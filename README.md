FILE structure:                
  - server.py — FastMCP server with 3 tools:                                                                            
    - ask(prompt) — runs Claude with SKILLS.md as system prompt, returns JSON                                         
    - ask_with_context(prompt, extra_context) — same but appends extra context to the system prompt                   
    - reload_skills() — returns the current SKILLS.md content
  - SKILLS.md — system prompt for the Claude agent (edit freely to change behavior)

  How it works:
  1. MCP client calls ask("your question")
  2. Server reads SKILLS.md → passes it as --system-prompt to claude -p --output-format json
  3. The JSON response from Claude is parsed and returned (code fences stripped automatically)

  Registered in ~/.claude.json as knowledge-mcp — restart Claude Code and it will appear in /mcp.

  To customize agent behavior, just edit SKILLS.md — no server restart needed (it's read on each call)