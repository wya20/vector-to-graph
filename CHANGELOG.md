# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-05-04

### Fixed

#### Core Graph Issues
- **Cross-file call edge extraction**: EnhancedTreeSitterParser now supports `global_func_ids` parameter to extract call edges across files
- **EnhancedTreeSitterParser not used**: API server now uses EnhancedTreeSitterParser instead of base TreeSitterParser
- **Graph search was vector search**: search_graph() now implements true BFS graph traversal, not just vector similarity
- **Graph data overwrites**: Index operation now incrementally merges nodes/edges instead of overwriting
- **Graph data lost on restart**: Server now restores graph from graph.json on startup via _restore_graph_from_disk()

#### Data Consistency
- **Chunker/Treesitter mismatch**: Server now reuses TreeSitterParser nodes to generate chunks, ensuring consistent boundaries
- **Merger discarding relations**: GraphResult.relations are now passed through metadata to build_context()

#### Integration Issues
- **TransactionLogger unused**: TransactionLogger now integrated into index_task for operation logging
- **FileWatcher unused**: Added /watch/{path} and /watch/stop endpoints to enable file watching
- **Weak tests**: Added test_cross_file_calls_edge() to verify call edge extraction

### Added

- **TreeSitterAlignedChunker class**: New chunker that reuses TreeSitter nodes for consistent chunk boundaries

## [0.2.0] - 2026-05-04

### Added

#### MCP AI 助手集成
- MCP Server (`src/mcp/server.py`) supporting 16+ AI assistants
- Supported platforms: Claude Code, Trae, Cursor, Windsurf, OpenCode, OpenClaw, Codex, CodeBuddy, Aider, Gemini CLI, and more
- `AGENTS.md` rule file for AI assistants
- `integrations/` directory with configuration examples for each platform

#### Core Functionality Fixes
- **TreeSitterParser Integration**: Fixed edges always being empty - graph now correctly shows function call relationships
- **Batch Encoding Optimization**: Changed from encoding one by one to batch processing - **5-10x performance improvement**
- **Graph Search Implementation**: Added `search_graph()` function, now truly performs graph search during queries
- **current_graph Update**: Saves graph data after indexing for subsequent queries

### Technical Updates
- Added `mcp[cli]>=1.0.0` dependency

### Supported File Types
- Python (.py)
- JavaScript (.js, .ts, .jsx, .tsx)
- Markdown (.md)

## [0.1.0] - 2026-04-26

### Added
- Initial release
- Vector search with Qdrant
- Knowledge graph with NetworkX
- Tree-sitter code parsing
- Community detection (Louvain/Leiden)
- REST API with FastAPI
