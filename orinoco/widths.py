import numpy as np
import networkx as nx
from tqdm import tqdm
from shapely.geometry import (LineString,
                              Point,
                              GeometryCollection,
                              MultiLineString,
                              )
from .rio_tools import get_geopandas_features_from_array
import geopandas as gpd
from shapely.ops import unary_union
import affine
from rasterio.transform import rowcol
from scipy.ndimage import find_objects
import scipy.ndimage as nd
from .nd_tools import apply_func_to_superpixels


##################
# Width Segments
##################


def get_width_features_from_segments(label_array: np.ndarray, profile: dict) -> np.ndarray:
    """Note:
    + assume label 0 is land and channel_mask is simply (label_array != 0)
    + width at label 0 will be set to np.nan
    + restricting calculations to bounding box of segment provides better approximation with respect to flow direction!
    + this is a rough approximation - the max in a segment is not quite right and using the distance transform does not
      consider flow direction, centroid position, channels lack of symmetry, etc.
    """
    transform = profile['transform']
    if transform.a != - transform.e:
        raise ValueError('Unequal x/y resolutions in channel mask and cannot use scipy')
    resolution = transform.a

    labels_unique = np.unique(label_array)
    indices = find_objects(label_array)

    m = len(labels_unique)
    width_features = np.zeros((m, 1))

    channel_mask = (label_array != 0)
    channel_dist_full = nd.distance_transform_edt(channel_mask) * resolution

    for k, label in enumerate(labels_unique):

        if label == 0:
            continue

        indices_temp = indices[label-1]

        # Buffer
        sy, sx = indices_temp
        sy = np.s_[max(sy.start - 1, 0): sy.stop + 1]
        sx = np.s_[max(sx.start - 1, 0): sx.stop + 1]
        indices_temp = sy, sx

        label_mask_in_slice = (label_array[indices_temp] == label)
        label_slice = label_array[indices_temp]
        channel_mask_slice = (label_slice != 0)

        # Our formula is: 2 * (nd.distance_transform_edt(channel_mask_slice)) - 1.
        # Note the "-1".
        # Because scipy determines distance using pixel's centers we overcount the 1/2 boundary pixels not in the channel
        # However, this could be an *underestimate* as along the diagonal this is \sqrt(2) / 2
        width_arr = 2 * (nd.distance_transform_edt(channel_mask_slice)) - 1
        width_arr *= resolution
        max_dist_in_label = np.nanmax(width_arr[label_mask_in_slice])

        # If no land in block, max_dist_in_label should be 0
        # To Ensure some positve ditance recorded, use full channel distance
        if max_dist_in_label == 0:
            channel_dist_full_slice = channel_dist_full[indices_temp]
            channel_dist_full_at_label = channel_dist_full_slice[label_slice == label]
            max_dist_in_label = np.nanmax(channel_dist_full_at_label)

        width_features[k] = max_dist_in_label

    width_features[0] = np.nan
    return width_features


def get_width_features_from_segments_naive(label_array: np.ndarray,  profile: dict) -> np.ndarray:
    """Note:
    + assume label 0 is land and channel_mask is simply (label_array == 0)
    + width at label 0 will be set to np.nan
    + uses the nd.distance_transform
    """
    transform = profile['transform']
    if transform.a != - transform.e:
        raise ValueError('Unequal x/y resolutions in channel mask and cannot use scipy')
    resolution = transform.a

    channel_mask = (label_array != 0).astype(np.uint8)

    channel_dist = nd.distance_transform_edt(channel_mask)
    # This could be an *underestimate* as this could be diagonal which would be approximately \sqrt(2) / 2
    width_features = (2 * apply_func_to_superpixels(np.nanmax, label_array, channel_dist) - 1) * resolution
    width_features[0] = np.nan
    return width_features


def add_width_features_to_graph(G: nx.classes.graph.Graph,
                                width_features: np.ndarray,
                                width_label: str = 'width_from_segment') -> nx.classes.graph.Graph:
    """Notes:
    + width_features indices should correspond to segment labels; stored as node attribute in G as "label"
    + width_features should be flattened (or `ravel()`-ed) so that they can be stored as numbers
    """
    node_data = dict(G.nodes(data=True))
    nodes = node_data.keys()

    def update_node_data(node):
        # label should correspond to the same feature
        label = node_data[node]['label']
        node_data[node][width_label] = width_features[label]

    list(map(update_node_data, nodes))
    nx.set_node_attributes(G, node_data)
    return G


##################
# Width Directions
# and Flows
##################


def unit_vector(vector: np.ndarray) -> np.array:
    """

    Parameters
    ----------
    vector : np.ndarray
        [TODO:description]

    Returns
    -------
    np.array:
        [TODO:description]
    """
    return vector / np.linalg.norm(vector)


def angle_between(v1: np.ndarray, v2: np.ndarray):
    """
    [TODO:summary]

    [TODO:description]

    Parameters
    ----------
    vector : np.ndarray
        [TODO:description]
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def edge_to_vector(edge_tuple: tuple):
    return np.array(edge_tuple[1]) - np.array(edge_tuple[0])


def realign_vector(edge: np.ndarray, reference_edge: np.ndarray):
    if angle_between(edge, reference_edge) <= np.pi/2:
        return edge
    else:
        return edge * -1


def _perp(t):
    return -t[1], t[0]


def get_vector_tail(center, direction, magnitude):
    return Point(center[0] + direction[0] * magnitude, center[1] + direction[1] * magnitude)


def _write_flow_perps(G, profile, out_name):
    """
    This is simply to inspect perpendicular flow vectors
    """
    grad_perps = nx.get_node_attributes(G, 'flow_vector_perp')
    nodes = list(grad_perps.keys())

    def get_line_string(node, ray_distance):
        grad_perp = grad_perps[node]
        transform = profile['transform']
        dx, dy = transform[0], -transform[4]
        p0 = (node[0] + grad_perp[0] * ray_distance * dx, node[1] + grad_perp[1] * ray_distance * dy)
        p1 = (node[0] - grad_perp[0] * ray_distance * dx, node[1] - grad_perp[1] * ray_distance * dy)
        return LineString([p0, p1])

    def get_line_string_partial(node):
        ray_distance = 20
        return get_line_string(node, ray_distance)

    geometry = list(map(get_line_string_partial, nodes))
    df = gpd.GeoDataFrame(geometry=geometry,
                          crs=dict(profile['crs']))
    df.to_file(out_name)
    return out_name


def get_flow_vector_from_gradient(node: tuple,
                                  width_dir_x: np.array,
                                  width_dir_y: np.array,
                                  transform: affine.Affine) -> np.ndarray:

    row, col = rowcol(transform, node[0], node[1])
    direction = np.array([width_dir_x[row, col], width_dir_y[row, col]])
    if np.isnan(direction[0]) | np.isnan(direction[1]):
        return None
    direction = direction / np.linalg.norm(direction)
    return tuple(direction)


def add_flow_attributes_from_gradient(G: nx.Graph,
                                      distance_arr: np.array,
                                      transform: affine.Affine):
    nodes = list(G.nodes())
    neg_nabla_y, nabla_x = np.gradient(distance_arr)
    width_x, width_y = neg_nabla_y, nabla_x

    def get_flow_vector_partial(node):
        return get_flow_vector_from_gradient(node, width_x, width_y, transform)
    grad_vectors_perp = list(map(get_flow_vector_partial, tqdm(nodes, desc='flow vector computation using gradient')))

    node_indices = list(range(len(nodes)))
    node_att_updates = {nodes[k]: {'flow_vector_perp_grad': grad_vectors_perp[k]} for k in node_indices}
    nx.set_node_attributes(G, node_att_updates)
    return G


def get_flow_vector_using_network(node: tuple, diG: nx.Graph,
                                  weighted_by_distance: bool = True,
                                  inv_dist_power: int = 1,
                                  clip_junction_neigbors: int = 2) -> np.ndarray:
    """
    Takes node and diG and obtain direction of flow at node dictated by network structure.

    Take a weighted average of neighboring vectors to determine which direction is most likely at
    given node. Weighted according to (1/distance) ** inv_dist_power.

    `clip_junction_neigbors` tells the function how many neighbors to use in the weighted sum.
    If None (or 0) all neighbors are used.
    """
    edge_neighbors = [(node, neighbor) for neighbor in nx.all_neighbors(diG, node)]
    neighbor_vectors = list(map(edge_to_vector, edge_neighbors))

    def realign_vectors_partial(edge_vector):
        return realign_vector(edge_vector, reference_edge=neighbor_vectors[0])
    neighbor_vectors = list(map(realign_vectors_partial, neighbor_vectors))

    neighbor_vectors_stacked = np.stack(neighbor_vectors, axis=0)

    norms = np.linalg.norm(neighbor_vectors_stacked, axis=1).reshape((neighbor_vectors_stacked.shape[0], -1))
    neighbor_unit_vectors_stacked = neighbor_vectors_stacked / norms

    if clip_junction_neigbors:
        indices = np.argsort(norms.ravel())
        norms = norms[indices][:clip_junction_neigbors]
        neighbor_unit_vectors_stacked = neighbor_unit_vectors_stacked[indices][:clip_junction_neigbors]

    if weighted_by_distance:
        weights = 1. / ((norms)**inv_dist_power)
        weights /= np.sum(weights)
        est_grad = np.sum(neighbor_unit_vectors_stacked * weights, axis=0)
    else:
        est_grad = np.mean(neighbor_unit_vectors_stacked, axis=0)
    return est_grad


def add_flow_attributes_from_network(G: nx.Graph):
    nodes = list(G.nodes())

    def get_flow_vector_partial(node):
        return get_flow_vector_using_network(node, G, weighted_by_distance=True)
    grad_vectors = list(map(get_flow_vector_partial, tqdm(nodes, desc='flow vector computation using network')))
    grad_vectors_perp = list(map(_perp, grad_vectors))
    node_indices = list(range(len(nodes)))
    node_att_updates = {nodes[k]: {'flow_vector_network': tuple(grad_vectors[k]),
                                   'flow_vector_perp_network': grad_vectors_perp[k]} for k in node_indices}
    nx.set_node_attributes(G, node_att_updates)
    return G


def add_flow_attributes(G, dist_arr, transform):
    G = add_flow_attributes_from_gradient(G, dist_arr, transform)
    G = add_flow_attributes_from_network(G)
    node_data = dict(G.nodes(data=True))

    def get_flow_perp(data):
        flow_perp = data.get('flow_vector_perp_grad')
        # if it's None, let's replace it
        flow_perp = flow_perp or data.get('flow_vector_perp_network')
        return flow_perp

    node_att_updates = {node: {'flow_vector_perp': get_flow_perp(data)}
                        for node, data in node_data.items()}
    nx.set_node_attributes(G, node_att_updates)
    return G

##################
# Width Geometries
##################


def get_segment_df(segments_arr, graph, profile):
    """
    Gets segment dataframe and appends data relevant for additional width computation
    """
    geo_features = get_geopandas_features_from_array(segments_arr.astype(np.int32),
                                                     profile['transform'],
                                                     mask=(segments_arr == 0),
                                                     connectivity=4)
    df_segments = gpd.GeoDataFrame.from_features(geo_features, crs=profile['crs'])

    node_data = dict(graph.nodes(data=True))
    # ensure we have flow directions and perps, perimeter, and other pertinent data
    # in geodataframe
    df_segments = update_segment_df_with_node_data(df_segments, node_data)
    return df_segments


def update_segment_df_with_node_data(df_segments, node_data):
    """
    perimieter, node (pos_x, pos_y), label, and flow_vector_perp (for widths)
    """
    df_segments = df_segments.dissolve(by='label').reset_index()

    bboxes = df_segments.geometry.envelope
    df_segments['perimeter'] = bboxes.exterior.length

    label_to_node = {data['label']: node for node, data in node_data.items()}
    label_to_flow_perp = {data['label']: data['flow_vector_perp']
                          for node, data in node_data.items()}

    df_segments['label'] = df_segments['label'].astype(np.int32)
    df_segments['node'] = df_segments['label'].map(lambda label: label_to_node.get(label, None))
    df_segments['flow_vector_perp'] = df_segments['label'].map(lambda label: label_to_flow_perp.get(label, None))

    return df_segments


def get_k_hop_neighborhood(graph, node, radius=1):
    current_layer = [node]
    final_neighborhood = [node]
    for k in range(radius):
        neighbors = [neighbor for node in current_layer for neighbor in nx.all_neighbors(graph, node)]
        neighbors = list(filter(lambda node: node not in final_neighborhood, neighbors))
        current_layer = neighbors.copy()
        final_neighborhood += neighbors
    return final_neighborhood


def buffer_segments_with_rag(graph, df_segments, radius=2):
    segment_to_geo_dict = df_segments.set_index('label').to_dict()['geometry']
    graph_ = graph.to_undirected()
    node_data = dict(graph_.nodes(data=True))
    neighbor_label_dict = {node_data[node]['label']: [node_data[neighbor]['label']
                                                      for neighbor in get_k_hop_neighborhood(graph, node, radius=radius)]
                           for node in node_data.keys()}

    def merge_neighbor_geometry(label):
        neighbors = neighbor_label_dict.get(label)
        if neighbors is None:
            return GeometryCollection()
        geometries = [segment_to_geo_dict[neighbor].buffer(1e-5)
                      for neighbor in neighbors]
        geometries += [segment_to_geo_dict[label].buffer(1e-5)]
        merged_geometries = (unary_union(geometries))
        return merged_geometries

    geom_series = df_segments.label.map(merge_neighbor_geometry)
    df_segment_buffered = gpd.GeoDataFrame({'label': df_segments.label.values},
                                           geometry=geom_series,
                                           crs=df_segments.crs)
    return df_segment_buffered


def get_candidate_line_df(df_segments):

    def get_line_string(row):
        direction = row['flow_vector_perp']
        length = row['perimeter'] / 2
        node = row['node']
        if node is not None:
            p0 = get_vector_tail(node, np.array(direction) * 1, length)
            p1 = get_vector_tail(node, np.array(direction) * -1, length)
            return LineString([p0, p1])
        else:
            return LineString()

    candidate_lines = df_segments.agg(get_line_string, axis=1)

    df_can = gpd.GeoDataFrame({'label': df_segments.label},
                              geometry=candidate_lines,
                              crs=df_segments.crs)
    return df_can


def get_geo_width_df(df_segments, graph, radius=2, geo_corrections=True):
    df_candidates = get_candidate_line_df(df_segments)
    df_segments_b = buffer_segments_with_rag(graph, df_segments, radius=radius)

    width_geometries = df_candidates.geometry.intersection(df_segments_b)
    df_widths = gpd.GeoDataFrame({'label': df_segments.label,
                                  'node': df_segments.node},
                                 geometry=width_geometries,
                                 crs=df_segments.crs
                                 )

    if geo_corrections:
        df_widths.geometry = df_widths.aggregate(_update_width_geometry, axis=1)

    df_widths['width_m'] = df_widths.geometry.length
    df_widths = df_widths[~df_widths.geometry.is_empty].copy()

    # Let's make sure after all that we have all the nodes accounted for
    assert(set(df_widths.node) == set(graph.nodes))
    return df_widths


def _update_width_geometry(row):
    """
    Basic geometric corrections:

    1. Ensures that if a width geometry contains a point, it only uses that point
    2. If the geometry does not contain a point, then it uses the full geometry
    3. If the geometry is a single point, return the empty linestring
    """
    width_geometry = row['geometry']
    node = row['node']

    width_line = width_geometry
    if isinstance(width_geometry, (MultiLineString, GeometryCollection)):
        def get_segment_with_node(line_seg):
            return line_seg.buffer(1).contains(Point(node))
        filtered_lines = list(filter(get_segment_with_node, width_line))
        if filtered_lines:
            width_line = filtered_lines[0]
    if isinstance(width_geometry, Point):
        width_line = LineString()
    return width_line


##################
# Final Width
# Determination
##################

def update_nodes_with_geometric_width_data(diG, width_df):
    df_dict = width_df.set_index('node').to_dict()
    widths_dict = df_dict['width_m']

    nodes = list(diG.nodes())
    node_att_updates = {node: {'width_m': widths_dict[node],
                               }
                        for node in nodes}
    nx.set_node_attributes(diG, node_att_updates)
    return diG


def update_edge_width_m(diG: nx.DiGraph):
    """
    Update edge widths as average width from nodes
    """

    # Get Node Data
    node_width_dict = nx.get_node_attributes(diG, 'width_m')
    if not node_width_dict:
        raise ValueError('diG must have `width_m` attribute')

    # Get Edge Data
    edge_data_ = (diG.edges(data=True))
    edge_data = {(e[0], e[1]): e[2] for e in edge_data_}
    edge_data.update({(e[1], e[0]): e[2] for e in edge_data_})
    edges = edge_data.keys()

    def get_edge_width(edge):
        width_0 = node_width_dict[edge[0]]
        width_1 = node_width_dict[edge[1]]
        return (width_0 + width_1) / 2
    edge_width_dict = {edge: {'width_m': get_edge_width(edge)} for edge in edges}
    nx.set_edge_attributes(diG, edge_width_dict)
    return diG


def update_graph_with_widths(diG: nx.DiGraph,
                             width_df):
    """
    Updates the nodes and edges according to the determinations above.
    """
    diG = update_nodes_with_geometric_width_data(diG, width_df)
    diG = update_edge_width_m(diG)
    return diG
