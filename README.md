# knowledge-mcp

A [FastMCP](https://gofastmcp.com) server that spawns a Claude CLI agent, grounds it in a local knowledge base, and returns structured JSON responses.

## How it works

1. An MCP client calls one of the tools below.
2. The server reads `skills/SKILLS.md` as the agent's system prompt.
3. It invokes `claude --print --output-format json` as a subprocess.
4. The JSON response is parsed and returned to the caller.

## Project layout

```
knowledge-mcp/
├── server.py          # FastMCP server — all tools defined here
├── skills/
│   └── SKILLS.md      # System prompt / skill definitions for the agent
├── knowledge/         # Drop your docs, papers, and kernels here
│   └── README.md      # Describes supported formats and suggested structure
└── pyproject.toml
```

## MCP tools

### General

| Tool | Description |
|---|---|
| `ask(prompt)` | Send a prompt to Claude using `SKILLS.md` as the system prompt. |
| `ask_with_context(prompt, extra_context)` | Same, but appends `extra_context` to the system prompt. |
| `reload_skills()` | Return the current contents of `skills/SKILLS.md`. |

### Knowledge base — BM25 retrieval

Python pre-processes the knowledge base: files are chunked (code split on `def`/`class`/`__global__` boundaries, prose by word-window with overlap), scored with BM25Okapi, and the top chunks are injected into the prompt before calling Claude.

| Tool | Description |
|---|---|
| `ask_knowledge(query, top_k=6)` | BM25-retrieve top chunks → pass as context → Claude answers with citations. Response includes `retrieved_sources`. |
| `search_knowledge(query, top_k=8)` | Return raw BM25 results without calling Claude — useful for inspecting coverage. |
| `list_knowledge()` | List every indexed file in `knowledge/` with its size. |

**When to use:** large corpora, predictable latency, lower token cost.

### Knowledge base — agent-driven retrieval

Claude itself explores the `knowledge/` folder using its built-in file tools (`Glob`, `Grep`, `Read`). No pre-chunking or scoring — the agent decides what to open and how deep to go.

| Tool | Description |
|---|---|
| `ask_knowledge_agent(query)` | Claude browses `knowledge/`, reads relevant files, synthesises an answer, and cites every source it used. |

**When to use:** small-to-medium corpora, queries that require following cross-references between files, or when you want the agent to judge relevance itself.

## Setup

```bash
# Requires Python >=3.10 and uv
uv sync

# Register with Claude Code

claude mcp remove knowledge-mcp -s user

claude mcp add --transport sse --scope user knowledge-mcp http://127.0.0.1:8000/sse
```



## Customising agent behaviour

Edit `skills/SKILLS.md` freely. Changes take effect on the next tool call — no server restart required.
