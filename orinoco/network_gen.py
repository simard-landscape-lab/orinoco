import networkx as nx
import numpy as np
from .nx_tools import get_RAG_neighbors, dfs_line_search
import skimage.measure as measure
from tqdm import tqdm
import scipy.ndimage as nd
from rasterio.transform import xy
from .nd_tools import get_features_from_array
from typing import Callable


def l2_difference(vec_1: tuple,
                  vec_2: tuple) -> float:
    """
    L^2 difference using tuples aka Euclidian distance.

    Parameters
    ----------
    vec_1 : tuple
        (x, y) in R^2
    vec_2 : tuple
        (x, y) in R^2

    Returns
    -------
    float:
        ||x - y||_2 for x, y in R^2
    """
    return np.linalg.norm(np.array(vec_1) - np.array(vec_2))


######################################
# Undirected Network
######################################


def get_undirected_channel_network(segment_labels: np.ndarray,
                                   dist: np.ndarray,
                                   profile: dict,
                                   interface_segment_labels: list,
                                   edge_distance_func: Callable = l2_difference,
                                   connectivity: int = 8) -> nx.classes.graph.Graph:
    """
    Obtain an undirected network from segments and distance array.

    Graph has keys that are positions (x, y) as UTM map coordinates.

    Parameters
    ----------
    segment_labels : np.ndarray
        m x n label array; labels must be from 1, 2, ..., n where we assume 0 is background/land.
    dist : np.ndarray
        m x n distance array (phi from fmm)
    profile : dict
        rasterio profile
    interface_segment_labels : list
        A list of segment labels that are adjacent to interface
    edge_distance_func : Callable
        A function f(a, b), where `a` and `b` are 2d tuples representing positions.
        In other words, a, b in R^2. Defaults to the normal L2 difference.
    connectivity : int
        4- or 8-connectivity for determining adjacency of derived pixes.
        4-connectivity means pixel neighbors must be in horizontally or vertically
        adjacenet. 8-connectivity adds diagaonal adjacency. Defaults to 8.
        See: https://en.wikipedia.org/wiki/Pixel_connectivity

    Returns
    -------
    nx.classes.graph.Graph:
        The graph from NetworkX to be returned. The graph has the following attributes:

        Nodes: (x, y)
            + label : segment label associated with node (int)
            + 'meters_to_interface' : distance determined using (x, y) position on dist (float)
            + x : x-position (float)
            + y : y-position (float)
            + interface_adj : adjacency to interface from FMM (bool)
        Edges: ((x1, y1), (x2, y2))
            + length: value from edge_distance_func, defaults to UTM (float)
            + weight: same as length (float)

        See NetworkX structure for details on reading such attributes.
    """

    # Obtain one distance within a segment - only need one since we thresholded them to generate segments
    distance_features = get_features_from_array(segment_labels, dist)
    distance_features = distance_features.ravel()

    transform = profile['transform']

    labels_unique = np.sort(np.unique(segment_labels))
    labels_unique = labels_unique[labels_unique > 0]

    # Computes
    props = measure.regionprops(segment_labels)

    # 0 is considered background label, props creaters array of length max labels.
    centroids_pixel_coords = [props[label - 1].centroid for label in labels_unique]
    centroids_map_coords = [transform * (c[1], c[0]) for c in centroids_pixel_coords]

    # Returns a list of lists indexed by the segment label.
    # Neigbors[label_0] = list of neighbor of label_0
    neighbors = get_RAG_neighbors(segment_labels, connectivity=connectivity)
    G = nx.Graph()

    # Make it easy to translate between centroids and labels
    label_to_centroid = {label: c for label, c in zip(labels_unique, centroids_map_coords)}
    centroid_to_label = {c: label for label, c in zip(labels_unique, centroids_map_coords)}

    def add_edge_to_graph(label):
        # Connect all labels to centroid
        edges_temp = [(label, connected_node) for connected_node in neighbors[label]]

        def get_edge_coords(edge_tuple):
            return label_to_centroid[edge_tuple[0]], label_to_centroid[edge_tuple[1]]

        edges_for_channel = list(map(get_edge_coords, edges_temp))
        G.add_edges_from(edges_for_channel)

    list(map(add_edge_to_graph, tqdm(labels_unique, desc='adding edges')))

    node_dictionary = {centroid: {'label': centroid_to_label[centroid],
                                  'meters_to_interface': distance_features[label],
                                  'x': centroid[0],
                                  'y': centroid[1],
                                  'interface_adj': (label in interface_segment_labels)}
                       for label, centroid in zip(labels_unique,
                                                  centroids_map_coords)
                       }
    edge_dictionary = {edge: {'length_m': edge_distance_func(*edge),
                              'weight': edge_distance_func(*edge)} for edge in G.edges()}

    nx.set_edge_attributes(G, edge_dictionary)
    nx.set_node_attributes(G, node_dictionary)

    return G


######################################
# Directed Network
######################################


def get_map_centroid_from_binary_mask(mask: np.ndarray,
                                      profile: dict) -> tuple:
    """
    Get map coordinates for a centroid from a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        m x n binary array.
    profile : dict
        rasterio profile associated with a map with dimensions and CRS of mask.

    Returns
    -------
    tuple:
        (x, y) position of centroid in map coordinates.
    """
    transform = profile['transform']
    ind_y, ind_x = nd.measurements.center_of_mass(mask.astype(np.uint8), [1])
    centroid = xy(transform, ind_y, ind_x)
    return centroid


def update_distance_using_graph_structure(G: nx.classes.graph.Graph,
                                          use_directed: bool = True) -> nx.Graph:
    """
    Because phi finds the shortest distance in the channel, we compute the distance along centerlines for
    a more accurate measurement.

    Parameters
    ----------
    G : nx.classes.graph.Graph
        G can be directed or undirected. The shortest path will be computed appropriately.
    use_directed : bool
        Can convert G to undirected for computing distances.
        Defaults to True. If G is undirected, then will be computed as expected.

    Returns
    -------
    nx.classes.graph.Graph:

        Overwrites attribute 'meters_to_interface' from Graph obtained via 'get_undirected_channel_network' or
        'direct_channel_network_using_distance'.
    """

    node_data = dict(G.nodes(data=True))
    nodes = list(node_data.keys())

    edge_data_ = (G.edges(data=True))
    edge_data = {(e[0], e[1]): e[2] for e in edge_data_}

    connected_to_interface = [node for node in nodes if (node_data[node]['interface_adj'])]

    # Obtain a proxy interface centroid for computing distance
    xs, ys = zip(*connected_to_interface)
    interface_centroid = np.mean(xs), np.mean(ys)

    G_aux = nx.DiGraph()

    edge_data_aux = edge_data.copy()
    edge_data_to_interface = {(node, interface_centroid): {'weight': 0,
                                                           'meters_to_interface': 0} for node in connected_to_interface}

    G_aux.add_edges_from(edge_data_aux.keys())
    G_aux.add_edges_from(edge_data_to_interface.keys())

    nx.set_edge_attributes(G_aux, edge_data_aux)
    nx.set_edge_attributes(G_aux, edge_data_to_interface)

    nx.set_node_attributes(G_aux, node_data)

    if use_directed:
        distance_dict = (nx.shortest_path_length(G_aux, target=interface_centroid, weight='weight'))
    else:
        distance_dict = (nx.shortest_path_length(G_aux.to_undirected(), target=interface_centroid, weight='weight'))

    # We are going to overwrite this graph attribute - no need to remove
    distance_dict_nodes = {node: {'meters_to_interface': distance_dict[node]}
                           for node in distance_dict.keys()}

    nx.set_node_attributes(G, distance_dict_nodes)
    nx.set_node_attributes(G_aux, distance_dict_nodes)

    return G


def reverse_line(line: list) -> list:
    """
    Takes a line of edges [(p0, p1), (p1, p2) ... (pn_m_1, pn)] and returns
    [(pn, pn_m_1), ... (p1, p0)], where the p's are (x, y) in map coordinates.

    Parameters
    ----------
    line : list
        List of edges (p0, p1), where p0, p1 in R^2

    Returns
    -------
    list:
        Returns list of edges reversed
    """
    return [(p1, p0) for (p0, p1) in reversed(line)]


def direct_line(line: list, node_data: dict) -> list:
    """
    This uses the nodata as obtained from `nx.get_node_attributes`
    and orients the edges of a line based on the attribute `meters_to_interface`.

    Orients edge towards interface, i.e. whichever endpoint is closer.

    Assumes line is provided as consecutive edges e.g. [(p0, p1), (p1, p2), ...]

    Parameters
    ----------
    line : list
       List of edges of the form (p, q), where p, q in R^2.
    node_data : dict
        Networkx node attribute dictionary of the form:
            node (key): data_dict (value)

        and data_dict is of the form:
            node_attribute_name (key):  attribute_value (value)

    Returns
    -------
    list:
        List of edges directed. May provivde both edge orientations if endpoints are
        equidistant to interface.
    """
    node_0 = line[0][0]
    node_1 = line[-1][-1]

    d_0 = node_data[node_0]['meters_to_interface']
    d_1 = node_data[node_1]['meters_to_interface']
    if (np.isnan(d_0)) or (np.isnan(d_1)):
        raise ValueError('meters to interface must be a number')

    n0_adj = node_data[node_0]['interface_adj']
    n1_adj = node_data[node_1]['interface_adj']

    # Both adjacent nodes means we put a multiedge that flows in both directions
    if n0_adj and n1_adj:
        return reverse_line(line) + line
    elif n0_adj:
        return reverse_line(line)
    elif n1_adj:
        return line
    elif d_0 > d_1:
        return line
    elif d_1 > d_0:
        return reverse_line(line)
    else:
        return reverse_line(line) + line


def filter_dangling_segments(G: nx.Graph, segment_threshold: int = 3, meters_to_interface: float = 100) -> nx.Graph:
    """
    This is a function to remove partitions of edges from graph one of whose endpoint is degree 1 and has
    size less than `segment_threshold`. Ignores such segments if at least one endpoint is:
        - less than or equal to `meters_to_interface` or
        - connected to the interface.

    The edge partition is determined by dividing edges between nodes that are junctions, adjacent to the interface, or
    have degree 1.

    Parameters
    ----------
    G : nx.Graph
        Uses the attributes `meters_to_interface` and `interface_adj`.
    segment_threshold : int
        The minimum number of edges in edge grouping to remain in the graph
    meters_to_interface : float
        If an endpoint of edge group is less than or equal to meters to interface, than do not remove.

    Returns
    -------
    nx.Graph:
        Changes input graph in place, but also returns object for clarity
    """

    node_data = dict(G.nodes(data=True))

    edge_data_ = (G.edges(data=True))
    edge_data = {(e[0], e[1]): e[2] for e in edge_data_}
    edges = list(edge_data.keys())

    if G.is_directed():
        G_und = G.to_undirected()
    else:
        G_und = G

    def filter_node_func(node):
        # Check if node is at least `meters_to_interface` meters from shore (accept dangling
        # segments if they are near ocean interface)
        # And has in degree 1 (leaf node)
        return ((G_und.degree(node) == 1)
                & (node_data[node]['meters_to_interface'] > meters_to_interface)
                & (not node_data[node]['interface_adj'])
                )

    def filter_edge_func(edge):
        # Check if edge belongs to  dangling segment
        node_0 = edge[0]
        node_1 = edge[1]
        return ((edge_data[edge]['edges_in_segment'] < segment_threshold) &
                ((filter_node_func(node_0)) |
                 (filter_node_func(node_1))
                 )
                )

    edges_within_seg_id = list(filter(filter_edge_func, edges))
    seg_ids_to_remove = [edge_data[edge]['segment_id'] for edge in edges_within_seg_id]

    G_filtered = nx.DiGraph()
    edges_filtered = [edge for edge in edges if edge_data[edge]['segment_id'] not in seg_ids_to_remove]
    G_filtered.add_edges_from(edges_filtered)

    nx.set_node_attributes(G_filtered, node_data)
    nx.set_edge_attributes(G_filtered, edge_data)
    return G_filtered


def direct_channel_network_using_distance(G: nx.Graph,
                                          remove_danlging_segments: bool = False,
                                          segment_threshold: int = 3,
                                          dangling_iterations: int = 1,
                                          meters_to_interface_filter_buffer: int = 100) -> nx.DiGraph:
    """
    Directs the channel network and adds additional graph attributes.

    Parameters
    ----------
    G : nx.Graph
        See `get_undirected_channel_network` for description.
    remove_danlging_segments : bool
        Pruning graph according to the parameters below
    segment_threshold : int
        Assume edge partition between junctions, degree 1 nodes, and those that touch the interface. Requires
        each edge grouping to have at least this size otherwise will be removed.
    dangling_iterations : int
        We can prune the graph iteratively (edge groups that were previously interior may now be admissable for pruning)
    meters_to_interface_filter_buffer : int
        We can ignore edge groups one of whose endpoints is within this distance to the interface.

    Returns
    -------
    nx.DiGraph:
        See `get_undirected_channel_network` for graph (node and edge) attributes in undirected case.
        These attributes are all added to the directed grap and updated appropriately.
        We additionally add:
        Nodes: (x, y)
            + graph_degree
        Edges: ((x1, y1), (x2, y2))
            + segment_id - unique integer id for edge grouping (int)
            + edges_in_segment - number of edges in particular edge group that edge belongs (int)
            + cc_id - unique integer id associated with different connected components (int)
    """
    diG = nx.DiGraph()

    # Get Node Data
    node_data = dict(G.nodes(data=True))
    nodes = list(node_data.keys())

    # Get Edge Data
    edge_data_ = (G.edges(data=True))
    edge_data = {(e[0], e[1]): e[2] for e in edge_data_}
    edge_data.update({(e[1], e[0]): e[2] for e in edge_data_})

    con_components = list(nx.connected_components(G))
    min_id = 0
    seg_id_data = {}

    for cc_id, component_nodes in enumerate(con_components):

        con_comps = G.subgraph(component_nodes)
        source = [node for node in con_comps if (node_data[node]['interface_adj'])][0]

        # Make sure that we partition
        def dfs_line_search_with_interface(G_con_comp, source):
            def interface_criterion(node):
                return node_data[node]['interface_adj']
            return dfs_line_search(G_con_comp, source, break_func=interface_criterion)

        lines = list(dfs_line_search_with_interface(con_comps, source))

        def direct_lines_partial(line):
            return direct_line(line, node_data)
        lines = list(map(direct_lines_partial, lines))
        list(map(diG.add_edges_from, lines))

        # Segment Ids
        data_for_seg = {edge:  {'segment_id': k + min_id,
                                'edges_in_segment': len(line),
                                'cc_id': cc_id}
                        for (k, line) in enumerate(lines)
                        for edge in line
                        }
        data_for_cc = {cc_node: {'cc_id': cc_id} for cc_node in component_nodes}

        min_id += len(lines)
        seg_id_data.update(data_for_seg)

    # Updating Attributes
    nx.set_node_attributes(diG, node_data)

    degree_nx = dict(G.degree)
    degree_dict = {node: {'graph_degree': degree_nx[node]} for node in nodes}
    nx.set_node_attributes(diG, degree_dict)
    nx.set_node_attributes(diG, data_for_cc)
    nx.set_edge_attributes(diG, edge_data)
    # This must come after edge data to update!
    nx.set_edge_attributes(diG, seg_id_data)

    diG = update_distance_using_graph_structure(diG)

    if remove_danlging_segments:
        for k in range(dangling_iterations):
            diG = filter_dangling_segments(diG,
                                           segment_threshold=segment_threshold,
                                           meters_to_interface=meters_to_interface_filter_buffer)
            diG = direct_channel_network_using_distance(diG.to_undirected(),
                                                        remove_danlging_segments=False,
                                                        segment_threshold=segment_threshold)
    return diG
