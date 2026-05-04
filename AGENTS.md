## Vector-to-Graph

This project uses Vector-to-Graph for hybrid search combining vector similarity and knowledge graph relationships.

Rules:
- Before answering code architecture or codebase questions, use the vector-to-graph MCP tools
- Use `vtg_query` to search instead of reading raw files when possible
- Use `vtg_get_graph` to explore code relationships (function calls, class hierarchies, imports)
- After modifying code files, run `vtg_index` to update the knowledge graph
- When asking about "how" or "why" code works, prefer graph relationships over simple text search

Available Tools:
- `vtg_index(path)`: Index a file or directory for search
- `vtg_query(question, limit)`: Query the hybrid search engine
- `vtg_get_graph()`: Get the knowledge graph structure
- `vtg_get_status()`: Check service status

Resources:
- `graph://data`: Access the knowledge graph data directly
