import json
import os
import re
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP
from rank_bm25 import BM25Okapi

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
SKILLS_FILE  = BASE / "skills" / "SKILLS.md"
KNOWLEDGE_DIR = BASE / "knowledge"

# file types to index in the knowledge base
INDEXED_EXTENSIONS = {".md", ".txt", ".rst", ".py", ".cu", ".cuh", ".c", ".cpp", ".triton"}

mcp = FastMCP("knowledge-mcp")


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
        return [], None

    tokenized = [c["text"].lower().split() for c in all_chunks]
    return all_chunks, BM25Okapi(tokenized)


def _search(query: str, top_k: int = 6) -> list[dict]:
    """Return the top_k most relevant chunks for query."""
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

    return results


def _run_claude_with_tools(prompt: str, system_prompt: str, extra_dirs: list[str] | None = None) -> dict:
    """Like _run_claude but keeps Claude's file tools enabled so it can read the knowledge dir."""
    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise RuntimeError("claude CLI not found in PATH")

    cmd = [
        claude_bin,
        "--print",
        "--output-format", "json",
        "--system-prompt", system_prompt,
        "--dangerously-skip-permissions",
    ]
    for d in (extra_dirs or []):
        cmd += ["--add-dir", d]
    cmd.append(prompt)

    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)

    if proc.returncode != 0:
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

    cmd = [
        claude_bin,
        "--print",
        "--output-format", "json",
        "--system-prompt", system_prompt,
        "--dangerously-skip-permissions",
        prompt,
    ]

    # Strip CLAUDECODE so the subprocess isn't blocked as a nested session
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)

    if proc.returncode != 0:
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
    return _run_claude(prompt, _load_skills())


@mcp.tool
def ask_with_context(prompt: str, extra_context: str) -> dict:
    """Send a prompt with additional context appended to the SKILLS.md system prompt."""
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
    )
    return _run_claude_with_tools(prompt, system_prompt, extra_dirs=[knowledge_dir])


@mcp.tool
def ask_knowledge(query: str, top_k: int = 6) -> dict:
    """
    Search the local knowledge base (knowledge/) for CUDA/Triton docs, papers,
    and code fragments most relevant to 'query', then pass them as grounded
    context to a Claude agent.  Returns a structured JSON answer with citations.
    """
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
        + "\n\nYou have access to a curated knowledge base of CUDA/Triton documentation, "
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
    mcp.run()
