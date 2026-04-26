import networkx as nx
from typing import List, Dict


class CommunityDetector:
    def __init__(self):
        self.graph = None

    def build_graph(self, nodes: List[dict], edges: List[dict]) -> nx.Graph:
        G = nx.Graph()
        for node in nodes:
            G.add_node(node['id'], **node.get('metadata', {}))
        for edge in edges:
            G.add_edge(edge['source'], edge['target'], **edge)
        self.graph = G
        return G

    def detect_communities(self, method: str = 'leiden') -> Dict[str, int]:
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph first.")

        if method == 'leiden':
            try:
                import graspologic
                from graspologic.algorithms import Leiden

                adj_matrix = nx.to_numpy_array(self.graph)
                leiden = Leiden()
                leiden.fit(adj_matrix)
                labels = leiden.predict()
                community_map = {}
                for idx, node in enumerate(self.graph.nodes()):
                    community_map[node] = int(labels[idx])
                return community_map
            except ImportError:
                pass

        import networkx.algorithms.community as nx_comm
        communities = nx_comm.louvain_communities(self.graph)
        community_map = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                community_map[node] = idx
        return community_map

    def get_community_stats(self, community_map: Dict[str, int]) -> List[dict]:
        from collections import Counter
        counts = Counter(community_map.values())
        return [
            {
                'community_id': k,
                'size': v,
                'nodes': [n for n in community_map if community_map[n] == k]
            }
            for k, v in counts.items()
        ]