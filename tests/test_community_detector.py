import pytest
from src.graph_builder.community_detector import CommunityDetector


class TestCommunityDetector:
    def setup_method(self):
        self.detector = CommunityDetector()

    def test_build_graph_empty(self):
        nodes = []
        edges = []
        G = self.detector.build_graph(nodes, edges)
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0

    def test_build_graph_with_nodes(self):
        nodes = [
            {'id': '1', 'metadata': {'label': 'A'}},
            {'id': '2', 'metadata': {'label': 'B'}}
        ]
        edges = [{'source': '1', 'target': '2'}]
        G = self.detector.build_graph(nodes, edges)
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 1
        assert '1' in G.nodes()
        assert '2' in G.nodes()

    def test_build_graph_node_metadata(self):
        nodes = [
            {'id': '1', 'metadata': {'label': 'A', 'type': 'test'}},
        ]
        edges = []
        G = self.detector.build_graph(nodes, edges)
        assert G.nodes['1']['label'] == 'A'
        assert G.nodes['1']['type'] == 'test'

    def test_build_graph_edge_attributes(self):
        nodes = [
            {'id': '1'},
            {'id': '2'}
        ]
        edges = [
            {'source': '1', 'target': '2', 'weight': 0.5}
        ]
        G = self.detector.build_graph(nodes, edges)
        assert G['1']['2']['weight'] == 0.5

    def test_detect_communities_empty_graph(self):
        self.detector.build_graph([], [])
        result = self.detector.detect_communities()
        assert result == {}

    def test_detect_communities_single_node(self):
        nodes = [{'id': '1'}]
        edges = []
        self.detector.build_graph(nodes, edges)
        result = self.detector.detect_communities()
        assert '1' in result

    def test_detect_communities_two_connected_nodes(self):
        nodes = [{'id': '1'}, {'id': '2'}]
        edges = [{'source': '1', 'target': '2'}]
        self.detector.build_graph(nodes, edges)
        result = self.detector.detect_communities()
        assert '1' in result
        assert '2' in result
        assert result['1'] == result['2']

    def test_detect_communities_multiple_communities(self):
        nodes = [
            {'id': '1'}, {'id': '2'}, {'id': '3'}, {'id': '4'}
        ]
        edges = [
            {'source': '1', 'target': '2'},
            {'source': '2', 'target': '1'},
            {'source': '3', 'target': '4'},
            {'source': '4', 'target': '3'},
        ]
        self.detector.build_graph(nodes, edges)
        result = self.detector.detect_communities()
        assert result['1'] == result['2']
        assert result['3'] == result['4']
        assert result['1'] != result['3']

    def test_get_community_stats_empty(self):
        community_map = {}
        stats = self.detector.get_community_stats(community_map)
        assert stats == []

    def test_get_community_stats_single_community(self):
        community_map = {'1': 0, '2': 0, '3': 0}
        stats = self.detector.get_community_stats(community_map)
        assert len(stats) == 1
        assert stats[0]['community_id'] == 0
        assert stats[0]['size'] == 3
        assert set(stats[0]['nodes']) == {'1', '2', '3'}

    def test_get_community_stats_multiple_communities(self):
        community_map = {'1': 0, '2': 0, '3': 1, '4': 1}
        stats = self.detector.get_community_stats(community_map)
        assert len(stats) == 2
        sizes = {s['community_id']: s['size'] for s in stats}
        assert sizes[0] == 2
        assert sizes[1] == 2