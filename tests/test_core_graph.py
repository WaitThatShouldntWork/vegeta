from vegeta.core.graph import Graph


def test_add_nodes_and_edges():
    g = Graph()
    g.add_node("A")
    g.add_node("B")
    g.add_edge("A", "B")
    assert g.neighbors("A") == ["B"]


