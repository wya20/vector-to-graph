# Vector-to-Graph MCP Integration

This directory contains integration configurations for various AI coding assistants.

## Supported Platforms

| Platform | Config File | Setup |
|----------|-------------|-------|
| Claude Code / Desktop | `claude-desktop/mcp.json` | `claude mcp add --transport http vtg http://localhost:8000/mcp` |
| Trae | `trae/mcp.json` | Copy to `.trae/mcp.json` |
| Cursor | `cursor/mcp.json` | Copy to `.cursor/mcp.json` |
| OpenCode | `opencode/mcp.json` | Copy to `.opencode/mcp.json` |
| OpenClaw | `openclaw/mcp.json` | Copy to `.claw/mcp.json` |
| Windsurf | `windsurf/mcp.json` | Copy to `.windsurf/mcp.json` |

## Quick Start

1. **Start the vector-to-graph service**:
   ```bash
   docker-compose up -d
   # or
   uvicorn src.api.server:app --reload --port 8000
   ```

2. **Start the MCP server**:
   ```bash
   uv run python -m src.mcp.server
   # or
   python -m src.mcp.server
   ```

3. **Connect your AI assistant**:

   For Claude Code:
   ```bash
   claude mcp add --transport http vtg http://localhost:8000/mcp
   ```

   For other platforms, copy the appropriate `mcp.json` to your platform's config directory.

## Available Tools

- `vtg_index(path)`: Index code/documents for hybrid search
- `vtg_query(question, limit)`: Query the hybrid search engine
- `vtg_get_graph()`: Get the knowledge graph structure
- `vtg_get_status()`: Check service status

## Resources

- `graph://data`: Access the knowledge graph directly
