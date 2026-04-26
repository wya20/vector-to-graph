import json
from pathlib import Path
from typing import List, Dict


class GraphExporter:
    def __init__(self, output_dir: str = "./data/graphs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, nodes: List[dict], edges: List[dict],
               community_map: Dict[str, int] = None,
               filename: str = "graph.json") -> str:
        graph_data = {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
        }
        if community_map:
            graph_data["community_map"] = community_map

        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        return str(output_path)

    def load(self, filename: str = "graph.json") -> dict:
        with open(self.output_dir / filename, 'r', encoding='utf-8') as f:
            return json.load(f)