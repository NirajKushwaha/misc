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

def get_neighbors_2d(lattice_size, node_2d_index, periodic=False):
    """
    Get neighboring node positions in a 2D square lattice.

    Parameters
    ----------
        lattice_size (int): Size N of the NxN square lattice.
        node_2d_index (tuple): (row, col) index of the node.
        periodic (bool): If True, uses periodic boundary conditions.

    Returns
    -------
        List[tuple]: List of (row, col) tuples of neighboring nodes.
    """
    
    N = lattice_size
    row, col = node_2d_index
    neighbors = []

    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in directions:
        nr, nc = row + dr, col + dc

        if periodic:
            nr %= N
            nc %= N
            neighbors.append((nr, nc))
        else:
            if 0 <= nr < N and 0 <= nc < N:
                neighbors.append((nr, nc))

    return neighbors