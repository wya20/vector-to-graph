from mcp.server.fastmcp import FastMCP
import httpx
from typing import Optional

mcp = FastMCP("Vector-to-Graph")

VTG_HOST = "http://localhost:8000"


async def call_vtg_api(endpoint: str, method: str = "GET", json_data: Optional[dict] = None) -> dict:
    url = f"{VTG_HOST}{endpoint}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        if method == "GET":
            response = await client.get(url)
        elif method == "POST":
            response = await client.post(url, json=json_data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def vtg_index(path: str, collection_name: Optional[str] = None) -> str:
    """Index code or documents for hybrid search.

    Args:
        path: Path to the file or directory to index
        collection_name: Optional collection name for Qdrant
    """
    try:
        data = {"path": path}
        if collection_name:
            data["collection_name"] = collection_name
        result = await call_vtg_api("/index", method="POST", json_data=data)
        return f"Indexing started: {result.get('message', 'OK')}"
    except Exception as e:
        return f"Index failed: {str(e)}"


@mcp.tool()
async def vtg_query(question: str, limit: int = 10) -> dict:
    """Query the hybrid search engine.

    Args:
        question: The question to search for
        limit: Maximum number of results to return
    """
    try:
        result = await call_vtg_api("/query", method="POST", json_data={
            "question": question,
            "limit": limit
        })
        return result
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def vtg_get_graph() -> dict:
    """Get the knowledge graph data including nodes and edges."""
    try:
        result = await call_vtg_api("/graph")
        return result
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def vtg_get_status() -> dict:
    """Get the current status of the vector-to-graph service."""
    try:
        result = await call_vtg_api("/status")
        return result
    except Exception as e:
        return {"error": str(e)}


@mcp.resource("graph://data")
async def get_graph_resource() -> dict:
    """Get the knowledge graph as a resource for AI context."""
    try:
        return await call_vtg_api("/graph")
    except Exception as e:
        return {"error": str(e)}


@mcp.prompt()
def query_template(question: str) -> str:
    """Generate a query prompt for architecture questions.

    Args:
        question: The user's question about the codebase
    """
    return f"""Answer the following question about the codebase using the vector-to-graph knowledge graph.

Question: {question}

Rules:
- First use vtg_query to search for relevant information
- Use vtg_get_graph to explore the knowledge graph structure
- Cite your sources with confidence scores
- Prefer graph relationships over raw file searches when available
- Synthesize information from both vector search and graph search for comprehensive answers
"""


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
