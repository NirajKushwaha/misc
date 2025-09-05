from .utils import *

def create_networkx_graph(network):
    """
    Create a networkx graph from an weighted edgelist.

    Parameters
    ----------
    network : dict
        Edgelist of the network.

    Returns
    -------
    nx.Graph
    """

    Graph = nx.Graph()

    for (u, v), weight in network.items():
        Graph.add_edge(u, v, weight=weight)

    return Graph