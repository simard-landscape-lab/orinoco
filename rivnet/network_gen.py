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
    return np.linalg.norm(np.array(vec_1) - np.array(vec_2))


######################################
# Undirected Network
######################################


def get_undirected_river_network(segment_labels: np.ndarray,
                                 dist: np.ndarray,
                                 profile: dict,
                                 interface_segment_labels: list,
                                 edge_distance_func: Callable = l2_difference) -> nx.classes.graph.Graph:
    """
    Note: `edge_ditance_func` should take two tuples and return a float.

    We use the l2_difference for the standard UTM distance. May use `get_meters_between_4326_points`
    as well (found in rio_tools) that uses geopy library.
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
    neighbors = get_RAG_neighbors(segment_labels)
    G = nx.Graph()

    # Make it easy to translate between centroids and labels
    label_to_centroid = {label: c for label, c in zip(labels_unique, centroids_map_coords)}
    centroid_to_label = {c: label for label, c in zip(labels_unique, centroids_map_coords)}

    def add_edge_to_graph(label):
        # Connect all labels to centroid
        edges_temp = [(label, connected_node) for connected_node in neighbors[label]]

        def get_edge_coords(edge_tuple):
            return label_to_centroid[edge_tuple[0]], label_to_centroid[edge_tuple[1]]

        edges_for_river = list(map(get_edge_coords, edges_temp))
        G.add_edges_from(edges_for_river)

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
    transform = profile['transform']
    ind_y, ind_x = nd.measurements.center_of_mass(mask.astype(np.uint8), [1])
    centroid = xy(transform, ind_y, ind_x)
    return centroid


def update_distance_using_graph_structure(G: nx.classes.graph.Graph,
                                          use_directed: bool = False,
                                          interface_centroid: tuple = None):

    node_data = dict(G.nodes(data=True))
    nodes = list(node_data.keys())

    edge_data_ = (G.edges(data=True))
    edge_data = {(e[0], e[1]): e[2] for e in edge_data_}

    connected_to_interface = [node for node in nodes if (node_data[node]['interface_adj'])]

    if interface_centroid is None:
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

    # We are going to overwrite this graph attribute - no need to remove :)
    distance_dict_nodes = {node: {'meters_to_interface': distance_dict[node]}
                           for node in distance_dict.keys()}

    nx.set_node_attributes(G, distance_dict_nodes)
    nx.set_node_attributes(G_aux, distance_dict_nodes)

    return G, G_aux


def reverse_line(line: list):
    return [(p1, p0) for (p0, p1) in reversed(line)]


def direct_line(line: list, node_data: dict):
    """
    Direct straight line path.

    We assume flow travels to interface
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


def filter_dangling_segments(G, segment_threshold=3, meters_to_interface=100):
    """
    Filters graph removing edges
     - with less than a threshold `min_edges_in_dangling_segment`
     - having a node with out degree 1
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
        return ((edge_data[edge]['edges_in_segment'] <= segment_threshold) &
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


def direct_river_network_using_distance(G: nx.Graph,
                                        remove_danlging_segments=False,
                                        segment_threshold=3,
                                        dangling_iterations=1,
                                        meters_to_interface_filter_buffer=100):
    """
    Uses `meter_to_coast` attribute and segmented graph to direct network
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

        G_con_comp = G.subgraph(component_nodes)
        source = [node for node in G_con_comp if (node_data[node]['interface_adj'])][0]

        # Make sure that we partition
        def dfs_line_search_with_interface(G_con_comp, source):
            def interface_criterion(node):
                return node_data[node]['interface_adj']
            return dfs_line_search(G_con_comp, source, break_func=interface_criterion)

        lines = list(dfs_line_search_with_interface(G_con_comp, source))

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

    diG, _ = update_distance_using_graph_structure(diG)

    if remove_danlging_segments:
        for k in range(dangling_iterations):
            diG = filter_dangling_segments(diG,
                                           segment_threshold=segment_threshold,
                                           meters_to_interface=meters_to_interface_filter_buffer)
            diG = direct_river_network_using_distance(diG.to_undirected(),
                                                      remove_danlging_segments=False,
                                                      segment_threshold=segment_threshold)
    return diG


######################################
# Add Widths
######################################

def add_widths_to_graph(G: nx.classes.graph.Graph, width_features: np.ndarray) -> nx.classes.graph.Graph:
    """Notes:
    + width_features indices should correspond to segment labels; stored as node attribute in G as "label"
    + width_features should be flattened (or `ravel()`-ed) so that they can be stored as numbers
    """
    node_data = dict(G.nodes(data=True))
    nodes = node_data.keys()

    def update_node_data(node):
        label = node_data[node]['label']
        node_data[node]['width'] = width_features[label]

    list(map(update_node_data, nodes))
    nx.set_node_attributes(G, node_data)
    return G
