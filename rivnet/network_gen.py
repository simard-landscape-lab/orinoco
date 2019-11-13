from .rio_tools import project_to_4326
import networkx as nx
import numpy as np
from .rio_tools import get_meters_between_points, _swap
from .nx_tools import get_RAG_neighbors, dfs_line_search
import skimage.measure as measure
import pyproj
from geopy import distance
from tqdm import tqdm
import scipy.ndimage as nd
from rasterio.transform import xy
from .nd_tools import get_features_from_array


######################################
# Undirected Network
######################################


def get_undirected_river_network(segment_labels: np.array, dist: np.array, profile: dict):

    # Obtain one distance within a segment - only need one since we thresholded them to generate segments
    distance_features = get_features_from_array(segment_labels, dist)
    distance_features = distance_features.ravel()

    transform = profile['transform']
    crs = str(profile['crs']).lower()
    source_proj = pyproj.Proj(init=crs)

    def project_partial(p): return project_to_4326(p, source_proj)

    labels_unique = np.sort(np.unique(segment_labels))
    labels_unique = labels_unique[labels_unique > 0]

    # Computes
    props = measure.regionprops(segment_labels)

    # 0 is considered background label, props creaters array of length max labels.
    centroids = [props[label - 1].centroid for label in labels_unique]
    centroids_src_cords = [transform * (c[1], c[0]) for c in centroids]
    centroids_lon_lat = list(map(project_partial, centroids_src_cords))

    # Returns a list of lists indexed by the segment label.
    # Neigbors[label_0] = list of neighbor of label_0
    neighbors = get_RAG_neighbors(segment_labels)
    G = nx.Graph()

    # Make it easy to translate between centroids and labels
    label_to_centroid = {label: c for label, c in zip(labels_unique, centroids_lon_lat)}
    centroid_to_label = {c: label for label, c in zip(labels_unique, centroids_lon_lat)}

    def add_edge_to_graph(label):
        # Connect all labels to centroid
        edges_temp = [(label, connected_node) for connected_node in neighbors[label]]

        def get_edge_coords(edge_tuple):
            return label_to_centroid[edge_tuple[0]], label_to_centroid[edge_tuple[1]]

        edges_for_river = list(map(get_edge_coords, edges_temp))
        G.add_edges_from(edges_for_river)

    list(map(add_edge_to_graph, labels_unique))

    node_dictionary = {centroid: {'label': centroid_to_label[centroid],
                                  'meters_to_coast': distance_features[label],
                                  'lon': centroid[0],
                                  'lat': centroid[1]}
                       for label, centroid, centroid_src in zip(labels_unique,
                                                                centroids_lon_lat,
                                                                centroids_src_cords)
                       }
    edge_dictionary = {edge: {'length_m': get_meters_between_points(*edge),
                              'weight': get_meters_between_points(*edge)} for edge in G.edges()}

    nx.set_edge_attributes(G, edge_dictionary)
    nx.set_node_attributes(G, node_dictionary)

    return G


######################################
# Directed Network
######################################


def get_latlon_centroid(mask, profile):
    transform = profile['transform']
    crs = str(profile['crs']).lower()
    source_proj = pyproj.Proj(init=crs)

    ind_y, ind_x = nd.measurements.center_of_mass(mask.astype(np.uint8), [1])
    x, y = xy(transform, ind_y, ind_x)
    lon, lat = project_to_4326((x, y), source_proj)
    centroid_latlon = lon, lat
    return centroid_latlon


def add_distance_to_ocean(G, ocean_centroid, use_directed=False):

    node_data = dict(G.nodes(data=True))
    nodes = list(node_data.keys())

    edge_data_ = (G.edges(data=True))
    edge_data = {(e[0], e[1]): e[2] for e in edge_data_}

    connected_to_sea = [node for node in nodes if (G.out_degree(node) == 0)]

    G_aux = nx.DiGraph()

    edge_data_aux = edge_data.copy()
    edge_data_to_ocean = {(node, ocean_centroid): {'weight': 0,
                                                   'meters_to_coast': 0} for node in connected_to_sea}

    G_aux.add_edges_from(edge_data_aux.keys())
    G_aux.add_edges_from(edge_data_to_ocean.keys())

    nx.set_edge_attributes(G_aux, edge_data_aux)
    nx.set_edge_attributes(G_aux, edge_data_to_ocean)

    nx.set_node_attributes(G_aux, node_data)

    if use_directed:
        distance_dict = (nx.shortest_path_length(G_aux, target=ocean_centroid, weight='weight'))
    else:
        distance_dict = (nx.shortest_path_length(G_aux.to_undirected(), target=ocean_centroid, weight='weight'))

    # We are going to overwrite this graph attribute - no need to remove :)
    distance_dict_nodes = {node: {'meters_to_coast': distance_dict[node]}
                           for node in distance_dict.keys()}

    nx.set_node_attributes(G, distance_dict_nodes)
    nx.set_node_attributes(G_aux, distance_dict_nodes)

    return G, G_aux


def get_linestring_length_in_meters(linestring):
    if linestring is None:
        return 0
    if linestring.is_empty:
        return 0
    else:
        p0, p1 = linestring.coords[0], linestring.coords[1]
        p0, p1 = _swap(p0), _swap(p1)
        return distance.distance(p0, p1).meters


def reverse_line(line: list):
    return [(p1, p0) for (p0, p1) in reversed(line)]


def direct_line(line, node_data):
    node_0 = line[0][0]
    node_1 = line[-1][-1]

    # if np.nan set d to 0
    d_0 = node_data[node_0]['meters_to_coast']
    d_1 = node_data[node_1]['meters_to_coast']
    if np.isnan(d_0):
        d_0 = 0
    if np.isnan(d_1):
        d_1 = 0

    if d_0 > d_1:
        return line
    elif d_1 > d_0:
        return reverse_line(line)
    else:
        return reverse_line(line) + line


def filter_dangling_segments(G, segment_threshold=3):
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
        # Check if node is at least 50 meters from shore (accept dangling
        # segments if they are near ocean interface)
        # And has in degree 1 (leaf node)
        return ((G_und.degree(node) == 1) &
                (node_data[node]['meters_to_coast'] > 50))

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


def direct_river_network_using_distance(G: nx.Graph, remove_danlging_segments=False):
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

    for cc_id, component_nodes in tqdm(enumerate(con_components)):

        G_con_comp = G.subgraph(component_nodes)

        source = min(component_nodes, key=lambda node: node_data[node]['meters_to_coast'])
        lines = list(dfs_line_search(G_con_comp, source))

        def direct_lines_partial(line): return direct_line(line, node_data)
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

    if remove_danlging_segments:
        diG = filter_dangling_segments(diG)
        diG = direct_river_network_using_distance(diG.to_undirected(), remove_danlging_segments=False)
    return diG
