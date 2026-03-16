import json
import logging
import os
import re
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP
from rank_bm25 import BM25Okapi

log = logging.getLogger("knowledge-mcp")
log.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
log.addHandler(_handler)
log.propagate = False  # don't feed into FastMCP's root handler chain

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
SKILLS_FILE  = BASE / "skills" / "SKILLS.md"
KNOWLEDGE_DIR = BASE / "knowledge"

# file types to index in the knowledge base
INDEXED_EXTENSIONS = {".md", ".txt", ".rst", ".py", ".cu", ".cuh", ".c", ".cpp", ".triton"}

mcp = FastMCP("knowledge-mcp")

# ── mcp library bug-fixes ───────────────────────────────────────────────────
# Two race-condition bugs exist in mcp 1.26.0 / fastmcp 3.1.1:
#
# Bug 1 (shared/session.py _receive_loop): when a request arrives before the
#   MCP initialise handshake completes, ServerSession._received_request raises
#   RuntimeError.  The exception is caught upstream and a JSON-RPC error is
#   sent, but the RequestResponder is left stuck in _in_flight with no cleanup.
#
# Bug 2 (shared/session.py _receive_loop notification branch): the subsequent
#   notifications/cancelled for that stale responder calls responder.cancel(),
#   which raises RuntimeError("RequestResponder must be used as a context
#   manager") because the responder was never __enter__-ed.
#
# Fix for Bug 1: override _received_request so that pre-init requests are
# rejected via a proper `with responder:` block (which calls on_complete and
# removes the responder from _in_flight) instead of raising.
#
# Fix for Bug 2: patch RequestResponder.cancel to silently handle the case
# where _entered is False (defensive belt-and-suspenders).

from mcp.server.session import ServerSession as _ServerSession, InitializationState as _InitState
from mcp.shared.session import RequestResponder as _RequestResponder
from mcp.types import (
    ErrorData as _ErrorData,
    INVALID_PARAMS as _INVALID_PARAMS,
    InitializeRequest as _InitializeRequest,
    PingRequest as _PingRequest,
)

_orig_received_request = _ServerSession._received_request


async def _patched_received_request(self, responder) -> None:
    """Reject pre-init requests cleanly (context manager + error response) instead of raising."""
    root = responder.request.root
    if (
        self._initialization_state != _InitState.Initialized
        and not isinstance(root, (_InitializeRequest, _PingRequest))
    ):
        log.debug(
            "[mcp-patch] rejecting pre-init request %s id=%s",
            type(root).__name__,
            responder.request_id,
        )
        with responder:
            await responder.respond(
                _ErrorData(
                    code=_INVALID_PARAMS,
                    message="Server not yet initialized",
                )
            )
        return
    await _orig_received_request(self, responder)


_ServerSession._received_request = _patched_received_request


_orig_cancel = _RequestResponder.cancel


async def _patched_cancel(self) -> None:
    """Cancel safely even if the responder was never entered as a context manager."""
    if not self._entered:
        self._completed = True
        try:
            self._session._in_flight.pop(self.request_id, None)  # type: ignore[attr-defined]
        except Exception:
            pass
        return
    await _orig_cancel(self)


_RequestResponder.cancel = _patched_cancel


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_skills() -> str:
    if SKILLS_FILE.exists():
        return SKILLS_FILE.read_text()
    return "You are a helpful assistant."


def _chunk_text(text: str, source: str, chunk_words: int = 300, overlap: int = 60) -> list[dict]:
    """
    Split text into word-windowed chunks with overlap.
    Code files are also split on top-level def/class/kernel boundaries first.
    """
    # For Python/CUDA code, try to split on function/class boundaries
    if source.endswith((".py", ".cu", ".cuh", ".c", ".cpp", ".triton")):
        blocks = re.split(r"\n(?=(?:def |class |__global__|__device__|__host__))", text)
        if len(blocks) > 1:
            chunks = []
            for block in blocks:
                chunks.extend(_chunk_text(block, source, chunk_words, overlap))
            return chunks

    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk_words_list = words[i : i + chunk_words]
        chunks.append({
            "text": " ".join(chunk_words_list),
            "source": source,
            "word_offset": i,
        })
        i += chunk_words - overlap
    return chunks


def _build_index() -> tuple[list[dict], Optional[BM25Okapi]]:
    """Walk KNOWLEDGE_DIR, chunk every file, build a BM25 index."""
    if not KNOWLEDGE_DIR.exists():
        log.warning("Knowledge directory not found: %s", KNOWLEDGE_DIR)
        return [], None

    all_chunks: list[dict] = []
    for file in sorted(KNOWLEDGE_DIR.rglob("*")):
        if file.is_file() and file.suffix.lower() in INDEXED_EXTENSIONS:
            try:
                text = file.read_text(errors="ignore")
                rel = str(file.relative_to(KNOWLEDGE_DIR))
                all_chunks.extend(_chunk_text(text, rel))
            except Exception:
                pass

    if not all_chunks:
        log.warning("No indexable files found in %s", KNOWLEDGE_DIR)
        return [], None

    log.info("Built BM25 index: %d chunks from %s", len(all_chunks), KNOWLEDGE_DIR)
    tokenized = [c["text"].lower().split() for c in all_chunks]
    return all_chunks, BM25Okapi(tokenized)


def _search(query: str, top_k: int = 6) -> list[dict]:
    """Return the top_k most relevant chunks for query."""
    log.info("Searching knowledge base — query=%r top_k=%d", query, top_k)
    chunks, bm25 = _build_index()
    if bm25 is None:
        return []

    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    seen_sources: dict[str, int] = {}
    for idx in ranked:
        if len(results) >= top_k:
            break
        c = chunks[idx]
        # At most 3 chunks per source to keep variety
        if seen_sources.get(c["source"], 0) < 3:
            results.append({**c, "score": round(float(scores[idx]), 4)})
            seen_sources[c["source"]] = seen_sources.get(c["source"], 0) + 1

    log.info("Search returned %d chunks: %s", len(results), [r["source"] for r in results])
    return results


def _run_claude_with_tools(prompt: str, system_prompt: str, extra_dirs: list[str] | None = None) -> dict:
    """Like _run_claude but keeps Claude's file tools enabled so it can read the knowledge dir."""
    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise RuntimeError("claude CLI not found in PATH")

    log.info("Invoking claude (with tools) — dirs=%s prompt_len=%d", extra_dirs, len(prompt))
    cmd = [
        claude_bin,
        "--print",
        "--output-format", "json",
        "--system-prompt", system_prompt,
        "--dangerously-skip-permissions",
    ]
    for d in (extra_dirs or []):
        cmd += ["--add-dir", d]

    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    # Pass the prompt via stdin so variadic args like --add-dir don't swallow it.
    proc = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )

    if proc.returncode != 0:
        log.error("claude (with tools) failed (exit %d): %s", proc.returncode, proc.stderr.strip())
        raise RuntimeError(f"claude exited {proc.returncode}: {proc.stderr.strip()}")

    raw = json.loads(proc.stdout)
    result_text = raw.get("result", "")
    stripped = result_text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        stripped = inner.strip()
    try:
        result_data = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        result_data = result_text

    log.info(
        "claude (with tools) completed — cost=$%.4f duration=%sms is_error=%s",
        raw.get("cost_usd") or 0,
        raw.get("duration_ms"),
        raw.get("is_error", False),
    )
    return {
        "result": result_data,
        "session_id": raw.get("session_id"),
        "cost_usd": raw.get("cost_usd"),
        "duration_ms": raw.get("duration_ms"),
        "is_error": raw.get("is_error", False),
    }


def _run_claude(prompt: str, system_prompt: str) -> dict:
    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise RuntimeError("claude CLI not found in PATH")

    log.info("Invoking claude — prompt_len=%d", len(prompt))
    cmd = [
        claude_bin,
        "--print",
        "--output-format", "json",
        "--system-prompt", system_prompt,
        "--dangerously-skip-permissions",
    ]

    # Strip CLAUDECODE so the subprocess isn't blocked as a nested session
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    proc = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )

    if proc.returncode != 0:
        log.error("claude failed (exit %d): %s", proc.returncode, proc.stderr.strip())
        raise RuntimeError(f"claude exited {proc.returncode}: {proc.stderr.strip()}")

    raw = json.loads(proc.stdout)

    # result_text may be JSON wrapped in ``` fences
    result_text = raw.get("result", "")
    stripped = result_text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        stripped = inner.strip()
    try:
        result_data = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        result_data = result_text

    log.info(
        "claude completed — cost=$%.4f duration=%sms is_error=%s",
        raw.get("cost_usd") or 0,
        raw.get("duration_ms"),
        raw.get("is_error", False),
    )
    return {
        "result": result_data,
        "session_id": raw.get("session_id"),
        "cost_usd": raw.get("cost_usd"),
        "duration_ms": raw.get("duration_ms"),
        "is_error": raw.get("is_error", False),
    }


def _format_context_block(chunks: list[dict]) -> str:
    """Render retrieved chunks into a readable context block for the agent."""
    if not chunks:
        return ""
    parts = ["## Retrieved Knowledge\n"]
    for i, c in enumerate(chunks, 1):
        parts.append(f"### [{i}] {c['source']} (score: {c['score']})\n```\n{c['text']}\n```\n")
    return "\n".join(parts)


# ── MCP tools ──────────────────────────────────────────────────────────────────

@mcp.tool
def ask(prompt: str) -> dict:
    """Send a prompt to a Claude agent using SKILLS.md as the system prompt.
    Returns the agent's response as structured JSON."""
    log.info("[ask] prompt=%r", prompt[:120])
    return _run_claude(prompt, _load_skills())


@mcp.tool
def ask_with_context(prompt: str, extra_context: str) -> dict:
    """Send a prompt with additional context appended to the SKILLS.md system prompt."""
    log.info("[ask_with_context] prompt=%r context_len=%d", prompt[:120], len(extra_context))
    system_prompt = _load_skills() + "\n\n## Additional Context\n" + extra_context
    return _run_claude(prompt, system_prompt)


@mcp.tool
def ask_knowledge_agent(query: str) -> dict:
    """
    Let the Claude agent autonomously explore the knowledge/ folder using its
    own file tools (Glob, Read, Grep) to find relevant CUDA/Triton docs, papers,
    and code, then answer the query.  No pre-chunking or BM25 — the agent decides
    what to read and how deep to go.
    """
    log.info("[ask_knowledge_agent] query=%r", query[:120])
    knowledge_dir = str(KNOWLEDGE_DIR)
    prompt = (
        f"The knowledge base is located at: {knowledge_dir}\n\n"
        f"## Task\n{query}\n\n"
    )
    system_prompt = (
        _load_skills()
        + "\n\nYou have full read access to a local knowledge base of CUDA "
          "documentation. Always read the files "
          "yourself before answering — do not guess from training data alone."
          "When searching for information, always read the index.json file in each folder first to identify relevant files. "
          "Only read files referenced in the index.json that are relevant to the user's query. Read less than 5 files"
    )
    return _run_claude_with_tools(prompt, system_prompt, extra_dirs=[knowledge_dir])


@mcp.tool
def ask_knowledge(query: str, top_k: int = 6) -> dict:
    """
    Search the local knowledge base (knowledge/) for CUDA docs, papers,
    and code fragments most relevant to 'query', then pass them as grounded
    context to a Claude agent.  Returns a structured JSON answer with citations.
    """
    log.info("[ask_knowledge] query=%r top_k=%d", query[:120], top_k)
    chunks = _search(query, top_k=top_k)
    context_block = _format_context_block(chunks)

    augmented_prompt = (
        f"{context_block}\n\n"
        f"## Question\n{query}\n\n"
        "Answer using the retrieved knowledge above. "
        "Cite source filenames in your answer."
    )

    system_prompt = (
        _load_skills()
        + "\n\nYou have access to a curated knowledge base of CUDA documentation, "
          "papers, and optimized kernel code. Ground your answers in the retrieved snippets "
          "and always cite the source."
    )

    result = _run_claude(augmented_prompt, system_prompt)
    result["retrieved_sources"] = [
        {"source": c["source"], "score": c["score"]} for c in chunks
    ]
    return result


@mcp.tool
def search_knowledge(query: str, top_k: int = 8) -> dict:
    """
    Search the knowledge base and return the top matching chunks without
    calling Claude — useful for inspecting what's available before asking.
    """
    log.info("[search_knowledge] query=%r top_k=%d", query[:120], top_k)
    chunks = _search(query, top_k=top_k)
    if not chunks:
        return {"chunks": [], "message": f"No documents found in {KNOWLEDGE_DIR}"}
    return {
        "query": query,
        "chunks": chunks,
        "total_returned": len(chunks),
        "knowledge_dir": str(KNOWLEDGE_DIR),
    }


@mcp.tool
def list_knowledge() -> dict:
    """List all files currently indexed in the knowledge base."""
    log.info("[list_knowledge] listing %s", KNOWLEDGE_DIR)
    if not KNOWLEDGE_DIR.exists():
        return {"files": [], "knowledge_dir": str(KNOWLEDGE_DIR)}

    files = [
        {"path": str(f.relative_to(KNOWLEDGE_DIR)), "size_kb": round(f.stat().st_size / 1024, 1)}
        for f in sorted(KNOWLEDGE_DIR.rglob("*"))
        if f.is_file() and f.suffix.lower() in INDEXED_EXTENSIONS
    ]
    return {"files": files, "total": len(files), "knowledge_dir": str(KNOWLEDGE_DIR)}


@mcp.tool
def reload_skills() -> dict:
    """Return the current contents of SKILLS.md."""
    return {"skills": _load_skills(), "path": str(SKILLS_FILE)}


if __name__ == "__main__":
    # mcp.run()
    mcp.run(transport="sse", host="127.0.0.1", port=8000)
