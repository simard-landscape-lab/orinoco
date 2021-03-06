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
from typing import Union

##################
# Width Segments
##################


def get_width_features_from_segments(label_array: np.ndarray,
                                     profile: dict) -> np.ndarray:
    """
    Takes a label array and rasterio profile (specifically its geotransform) to
    obtain width using the (scipy) distance transform in a neighborhood of each
    segment.

    The width within a segment is computed as `(2*d - 1) * res`, where `d` is
    the maximum of the distance transform in that segment where we compute the
    distance transform in a small 1 pixel buffered bounding box of the segment
    and the `res` is the resolution (in meters; we assume all rasters are in
    meters) determined using the rasterio profile.

    We assume label 0 is land and the channel_mask is (label_array != 0).

    The `get_width_features_from_segments_naive` uses the distance transform on
    the full channel mask rather than 1 pixel buffered bounding box of the
    segment.

    Parameters
    ----------
    label_array : np.ndarray
        array of labels (m x n) with p unique labels
    profile : dict
        Rasterio profile dictionary

    Returns
    -------
    np.ndarray:
        Obtain features of shape (p x 1) where `p:= unique labels in label
        array` and index i of feature vector corresponds to width of label i.

    Notes
    -----
    + width at label 0 will be set to np.nan
    + If distance transform at particular segment is 0 then we assign width at
    node the width determined using the full distance transform determined
    using the channel mask, i.e.  (label_array != 0) not just the distance
    transfom in a buffered area around the segment.
    """
    transform = profile['transform']
    if transform.a != - transform.e:
        msg = 'Unequal x/y resolutions in channel mask and cannot use scipy'
        raise ValueError(msg)
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

        # Our formula is: 2 * (nd.distance_transform_edt(channel_mask_slice)) -
        # 1.  Note the "-1".  Because scipy determines distance using pixel's
        # centers we overcount the 1/2 boundary pixels not in the channel
        # However, this could be an *underestimate* as along the diagonal this
        # is \sqrt(2) / 2
        width_arr = 2 * (nd.distance_transform_edt(channel_mask_slice)) - 1
        width_arr *= resolution
        max_dist_in_label = np.nanmax(width_arr[label_mask_in_slice])

        # If no land in block, max_dist_in_label should be 0
        # To Ensure some positve ditance recorded, use full channel distance
        if max_dist_in_label == 0:
            channel_dist_full_slice = channel_dist_full[indices_temp]
            temp_slice = label_slice == label
            channel_dist_full_at_label = channel_dist_full_slice[temp_slice]
            max_dist_in_label = np.nanmax(channel_dist_full_at_label)

        width_features[k] = max_dist_in_label

    width_features[0] = np.nan
    return width_features


def get_width_features_from_segments_naive(label_array: np.ndarray,
                                           profile: dict) -> np.ndarray:
    """
    Takes a label array and obtains width using the distance transform using
    the entire channel mask (label_array != 0). Specifically, the width within
    a segment is determined as `(2*d - 1) * res`, where `d` is the distance
    transform determined using the the entire channel mask. Assume label 0 is
    land and channel_mask is (label_array != 0).

    Contrast this with get_width_features_from_segments which computes distance
    transform only within a buffered bounding box of the segment.

    Parameters
    ----------
    label_array : np.ndarray
        array of labels (m x n)
    profile : dict
        Rasterio profile dictionary

    Returns
    -------
    np.ndarray:
        Obtain features of shape (p x 1) where p:= unique labels in label array
        and index i of feature vector corresponds to width of label i.

    Notes
    -----
    + width at label 0 will be set to np.nan
    """
    transform = profile['transform']
    if transform.a != - transform.e:
        msg = 'Unequal x/y resolutions in channel mask and cannot use scipy'
        raise ValueError(msg)
    resolution = transform.a

    channel_mask = (label_array != 0).astype(np.uint8)

    channel_dist = nd.distance_transform_edt(channel_mask)
    # This could be an *underestimate* as this could be diagonal which would be
    # approximately \sqrt(2) / 2
    d = apply_func_to_superpixels(np.nanmax, label_array, channel_dist)
    width_features = (2 * d - 1) * resolution
    width_features[0] = np.nan
    return width_features


def add_width_features_to_graph(G: nx.classes.graph.Graph,
                                width_features: np.ndarray,
                                width_label: str = 'width_from_segment')\
                                        -> nx.Graph:
    """
    Take width features of length p in which index i has width at that label
    and add that as node attribute for node with label i.  Width features
    should be flattened (or width_features.ravel()) We take the node_data =
    {node: data_dict for node in G.nodes()} and update each data_dict with
    `width_label`: width.

    Parameters
    ----------
    G : nx.classes.graph.Graph
        Graph to update
    width_features : np.ndarray
        Features of widths with index i corresponding to width at label i
    width_label : str
        Label to update node attribute. Defaults to `width_from_segment`. Keep
        default to ensure other analyses work as expected without modification.

    Returns
    -------
    nx.Graph:
        Graph is modified in place and returned
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
    Normalize a vector to have unit l2 norm

    Parameters
    ----------
    vector : np.ndarray

    Returns
    -------
    np.array:
        Normalized vector, i.e. v / ||v||_2
    """
    return vector / np.linalg.norm(vector)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Find the angle between two vectors using the arccosine. Specifically,
    arcos( v1 * v2 ) / (||v1||_2 ||v2||_2), where * indicates the vector dot
    product.

    Parameters
    ----------
    v1 : np.ndarray
        Vector
    v2 : np.ndarray
        Vector

    Returns
    -------
    float:
        The angle in radians
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def edge_to_vector(edge_tuple: tuple) -> np.ndarray:
    """
    Transforms edges of the form (p0, p1) to vector p1 - p0,
    where p0, p1 are 2d vectors.

    Parameters
    ----------
    edge_tuple : tuple
        Tuple of tuples, i.e. (p0, p1), where p0, p1 in R2.

    Returns
    -------
    np.ndarray:
        Edge vector indicating direction of edge from p0.
    """
    return np.array(edge_tuple[1]) - np.array(edge_tuple[0])


def realign_vector(edge: np.ndarray, reference_edge: np.ndarray) -> np.ndarray:
    """
    Using the reference_edge, ensure that the other edge is within pi/2. If
    not, take its negative aka reverse the vector aka reflect through origin.

    Parameters
    ----------
    edge : np.ndarray
        Edge to possibly reverse
    reference_edge : np.ndarray
        Edge that remains fixed

    Returns
    -------
    np.ndarray:
        edge or -edge depending on angle between.
    """
    if angle_between(edge, reference_edge) <= np.pi/2:
        return edge
    else:
        return edge * -1


def _perp(t: tuple) -> tuple:
    """
    Obtain the point whose corresponding vector is perpendicular to t.

    Parameters
    ----------
    t : tuple
       Point (x, y)

    Returns
    -------
    tuple:
        (-y, x)
    """
    return -t[1], t[0]


def get_vector_tail(center: tuple,
                    direction: tuple,
                    magnitude: float) -> LineString:
    """
    Obtain a LineString of the (center, center + direction * magnitude)

    Assume direction is unit vector.

    Parameters
    ----------
    center : tuple
        Head of LineString
    direction : tuple
        Direction from center; assume is unit norm.
    magnitude : float
        Length of desired output vector

    Returns
    -------
    LineString:
        Shaply geometry of the vector
    """
    return Point(center[0] + direction[0] * magnitude,
                 center[1] + direction[1] * magnitude)


def _lookup_flow_vector_from_widths_arr(node: tuple,
                                        width_x: np.array,
                                        width_y: np.array,
                                        transform: affine.Affine)\
                                                -> Union[np.ndarray, None]:
    """
    Obtain the unit vector direction from an array of widths in x and y
    directions.  Here, this will be the perpindicular line to  the gradient of
    distance function determined with fmm.

    We expect the following for the width arrays:

        neg_nabla_y, nabla_x = np.gradient(distance_arr)
        width_x, width_y = neg_nabla_y, nabla_x

    Parameters
    ----------
    node : tuple
        (x, y) in R2 in map coordinates associated with transform below
    width_x : np.array
        Will be the negative of nabla_y
    width_y : np.array
        Will be nabla_x
    transform : affine.Affine
        The rasterio transform associated with the width_x and width_y arrays.

    Returns
    -------
    np.ndarray or None:
       2d unit vector indicating width direction. Returns none if gradient is
       np.nan
    """

    row, col = rowcol(transform, node[0], node[1])
    direction = np.array([width_x[row, col], width_y[row, col]])
    if np.isnan(direction[0]) | np.isnan(direction[1]):
        return None
    direction = direction / np.linalg.norm(direction)
    return tuple(direction)


def add_flow_attributes_from_gradient(G: nx.Graph,
                                      distance_arr: np.array,
                                      transform: affine.Affine) -> nx.Graph:
    """
    Using the distance array and the associated transform of this array (see
    rasterio), update a graph with the node attribute `flow_vector_perp_grad`
    which is perpendicular to the the gradient of the distance_arr where we
    assume that the coordinates are in the bottom left corner. If the node is
    not within the channel (i.e. has no gradient defined), we return None

    Parameters
    ----------
    G : nx.Graph
        Graph to update
    distance_arr : np.array
        Distance array from the fast marching method (phi)
    transform : affine.Affine
        The transform (see rasterio) associated with distance_arr

    Returns
    -------
    nx.Graph:
        The graph updated in place is returned
    """
    nodes = list(G.nodes())
    # np.gradient takes the gradient with respect to the upper left corner
    neg_nabla_y, nabla_x = np.gradient(distance_arr)
    width_x, width_y = neg_nabla_y, nabla_x

    def get_flow_perp_vector_partial(node):
        return _lookup_flow_vector_from_widths_arr(node,
                                                   width_x,
                                                   width_y,
                                                   transform)
    desc = 'flow vector computation using gradient'
    grad_vectors_perp = list(map(get_flow_perp_vector_partial,
                                 tqdm(nodes,
                                      desc=desc)))

    node_indices = list(range(len(nodes)))
    node_att_updates = {nodes[k]: {'flow_vector_perp_grad':
                                   grad_vectors_perp[k]}
                        for k in node_indices}
    nx.set_node_attributes(G, node_att_updates)
    return G


def _get_flow_vector_using_network(node: tuple,
                                   G: nx.Graph,
                                   weighted_by_inv_distance: bool = True,
                                   inv_dist_power: int = 1,
                                   top_n_neighbors: int = None) -> np.ndarray:
    """
    Uses the network structure to obtain a gradient. For each node, takes the
    weighted average of all the directions determined by edges (ensuring they
    are all in the same quadrant) with weights determined by their inverse
    distance to the node under consideration. Let p = inv_dist_power. This
    parameter allows us to specify weights as 1 / ||x||**p.  Also, if n =
    top_n_neighbors, when n is not None, we only consider the top n closest
    neighbors in the average.

    Parameters
    ----------
    node : tuple
        The node (x, y) under consideration.
    diG : nx.Graph
        The graph the node lives in. For looking up neighbors.
    weighted_by_inv_distance : bool
        This means do we consider the normal mean or weighted by inverse
        distance. Defaults to True.
    inv_dist_power : int
        The power in the weighting of inverse distance. Defaults to 1.
    top_n_neighbors : int
        If not None or 0, then we clip the top closest neighbors for averaging
        of edges and flow direction.

    Returns
    -------
    np.ndarray:
        Returns 2d vector of estimated gradient using the edges emanating from
        node.  This will be a unit vector.
    """
    edge_neighbors = [(node, neighbor)
                      for neighbor in nx.all_neighbors(G, node)]
    neighbor_vectors = list(map(edge_to_vector, edge_neighbors))

    def realign_vectors_partial(edge_vector):
        return realign_vector(edge_vector, reference_edge=neighbor_vectors[0])
    neighbor_vectors = list(map(realign_vectors_partial, neighbor_vectors))

    neighbor_vectors_stacked = np.stack(neighbor_vectors, axis=0)

    norms = np.linalg.norm(neighbor_vectors_stacked, axis=1)
    norms = norms.reshape((neighbor_vectors_stacked.shape[0], -1))
    neighbor_vectors_u_stacked = neighbor_vectors_stacked / norms

    if top_n_neighbors:
        indices = np.argsort(norms.ravel())
        n = top_n_neighbors
        norms = norms[indices][:n]
        neighbor_vectors_u_stacked = neighbor_vectors_u_stacked[indices][:n]

    if weighted_by_inv_distance:
        weights = 1. / ((norms)**inv_dist_power)
        weights /= np.sum(weights)
        est_grad = np.sum(neighbor_vectors_u_stacked * weights,
                          axis=0)
    else:
        est_grad = np.mean(neighbor_vectors_u_stacked, axis=0)
    est_grad = est_grad / np.linalg.norm(est_grad)
    return est_grad


def add_flow_attributes_from_network(G: nx.Graph) -> nx.Graph:
    """
    Updates the Graph G with two new node attributes using the edge structure.
    Namely, it adds: `flow_vector_network` and `flow_vector_perp_network`.

    Parameters
    ----------
    G : nx.Graph
        Graph to update. Updated in place.

    Returns
    -------
    nx.Graph:
        Updates graph with two node attributes: `flow_vector_network` and
        `flow_vector_perp_network`.
    """
    nodes = list(G.nodes())

    def get_flow_vector_partial(node):
        return _get_flow_vector_using_network(node, G,
                                              weighted_by_inv_distance=True)
    desc = 'flow vector computation using network'
    grad_vectors = list(map(get_flow_vector_partial,
                            tqdm(nodes, desc=desc)))
    grad_vectors_perp = list(map(_perp, grad_vectors))
    node_indices = list(range(len(nodes)))
    node_att_updates = {nodes[k]: {'flow_vector_network':
                                   tuple(grad_vectors[k]),
                                   'flow_vector_perp_network':
                                   grad_vectors_perp[k]} for k in node_indices}
    nx.set_node_attributes(G, node_att_updates)
    return G


def add_flow_attributes(G: nx.Graph,
                        dist_arr: np.ndarray,
                        transform: affine.Affine) -> nx.Graph:
    """
    Adds flow attributes from gradient and network. If gradient at a node is
    defined use this for `flow_vector_perp`, otherwise use network defined
    geometry. Graph will have new node attribute `flow_vector_perp`.

    Parameters
    ----------
    G : nx.Graph
        Graph to be updated in place.
    dist_arr : np.ndarray
        Distance array from fast marching method (phi).
    transform : affine.Affine
        Transform associated with dist_arr (see rasterio).

    Returns
    -------
    nx.Graph:
        Graph updated in place with new node attribute `flow_vector_perp`.
    """
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


def get_segment_df(segments_arr: np.ndarray,
                   G: nx.Graph, profile: dict,
                   connectivity: int = 4) -> gpd.GeoDataFrame:
    """
    Gets segment dataframe and appends data relevant for additional width
    computation. This assumes that G has been updated using
    `add_flow_attributes`. We include the following columns in our dataframe
    for later analysis:

        perimieter, node (pos_x, pos_y), label, and flow_vector_perp

    Parameters
    ----------
    segments_arr : np.ndarray
        Segment labels (m x n)
    G : nx.Graph
        RAG graph associated with segments
    profile : dict
        Rasterio profile associated with segment_arr
    connectivity : int
        4 or 8 connectivity accepted.

        See: https://en.wikipedia.org/wiki/Pixel_connectivity

    Returns
    -------
    gpd.GeoDataFrame:
        A dataframe in which each geometry is a polygon associated with a
        contigious segment. We also include perimieter, node (pos_x, pos_y),
        label, and flow_vector_perp (for widths)

    Note
    ----
    Not every segment may have a node and corresponding node attributes because
    we have pruned them previously.  We remove those nodes that are not apart
    of G.
    """
    segments_arr = segments_arr.astype(np.int32)
    geo_features = get_geopandas_features_from_array(segments_arr,
                                                     profile['transform'],
                                                     mask=(segments_arr == 0),
                                                     connectivity=connectivity)
    df_segments = gpd.GeoDataFrame.from_features(geo_features,
                                                 crs=profile['crs'])

    node_data = dict(G.nodes(data=True))
    # ensure we have flow directions and perps, perimeter, and other pertinent
    # data in geodataframe
    df_segments = _update_segment_df_with_node_data(df_segments, node_data)
    return df_segments


def _update_segment_df_with_node_data(df_segments: gpd.GeoDataFrame,
                                      node_data: dict) -> gpd.GeoDataFrame:
    """
    Adds columns:

        perimieter, node (pos_x, pos_y), label, and flow_vector_perp

    to segment geodataframe.

    Parameters
    ----------
    df_segments : gpd.GeoDataFrame
        The segment dataframe to be modified in place with new columns
    node_data : dict
        The node_data dict obtained from dict(G.nodes(data=true))

    Returns
    -------
    gpd.GeoDataFrame:
        df_segments modified
    """
    # This ensures that if multiple polygons were required to specify a single
    # label That they are merged into 1 row - this is true when we use
    # 8-connectivity and a polygon is connected through a diagonal.
    df_segments = df_segments.dissolve(by='label').reset_index()

    bboxes = df_segments.geometry.envelope
    df_segments['perimeter'] = bboxes.exterior.length

    label_to_node = {data['label']: node for node, data in node_data.items()}
    label_to_flow_perp = {data['label']: data['flow_vector_perp']
                          for node, data in node_data.items()}

    df_segments['label'] = df_segments['label'].astype(np.int32)

    def node_lookup(label):
        return label_to_node.get(label, None)
    df_segments['node'] = df_segments['label'].map(node_lookup)

    def flow_lookup(label):
        return label_to_flow_perp.get(label, None)
    df_segments['flow_vector_perp'] = df_segments['label'].map(flow_lookup)

    return df_segments


def get_k_hop_neighborhood(G: nx.Graph, node: tuple, radius: int = 1) -> list:
    """
    Obtain the nodes within a radius of a node within G (ignoring directivity)
    and without requiring the creation of a new graph object. Same output (but
    faster) of `nx.ego_graph(G, node, radius=radius).nodes()` because doesn't
    create a new graph object.

    Parameters
    ----------
    G : nx.Graph
    node : tuple
    radius : int
        The number of "hops" allowed to obtain nodes from `G` around `node`.

    Returns
    -------
    list:
        List of nodes
    """
    current_layer = [node]
    final_neighborhood = [node]
    for k in range(radius):
        neighbors = [neighbor for node in current_layer
                     for neighbor in nx.all_neighbors(G, node)]

        def filter_func(node): return node not in final_neighborhood
        neighbors = list(filter(filter_func, neighbors))
        current_layer = neighbors.copy()
        final_neighborhood += neighbors
    return final_neighborhood


def buffer_segments_with_rag(G: nx.Graph,
                             df_segments: gpd.GeoDataFrame,
                             radius: int = 1) -> gpd.GeoDataFrame:
    """
    Creates a copy of df_segments in which geometries are buffered by neighbors
    of specified radius (determined by Graph G/RAG).

    Parameters
    ----------
    G : nx.Graph
        Graph associated with df_segments
    df_segments : gpd.GeoDataFrame
        Dataframe with segments and related data with polygons representing
        areas of segment labels
    radius : int
        Number of hops to buffer individual segment. Default 1.

    Returns
    -------
    gpd.GeoDataFrame:
        Same shape as df_segments with buffered geometry according to G and
        radius (if radius = k, we consider the k-hop neighborhood of segment
        within G).
    """
    segment_to_geo_dict = df_segments.set_index('label').to_dict()['geometry']
    graph_ = G.to_undirected()
    node_data = dict(graph_.nodes(data=True))
    r = radius
    neighbor_label_dict = {node_data[node]['label']:
                           [node_data[neighbor]['label']
                           for neighbor in get_k_hop_neighborhood(G,
                                                                  node,
                                                                  radius=r)]
                           for node in node_data.keys()}

    def merge_neighbor_geometry(label):
        neighbors = neighbor_label_dict.get(label)
        if neighbors is None:
            return GeometryCollection()
        geometries = [segment_to_geo_dict[neighbor].buffer(1e-9)
                      for neighbor in neighbors]
        geometries += [segment_to_geo_dict[label].buffer(1e-9)]
        merged_geometries = (unary_union(geometries))
        return merged_geometries

    geom_series = df_segments.label.map(merge_neighbor_geometry)
    df_segment_buffered = gpd.GeoDataFrame({'label': df_segments.label.values},
                                           geometry=geom_series,
                                           crs=df_segments.crs)
    return df_segment_buffered


def get_candidate_line_df(df_segments: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Obtain a dataframe of same height (axis=0) as df_segments with labels and
    geometry a LineString that will be an upper bound for final measured width
    geometry. We specify an upper bounder using the perimeter of the segment
    and define the width geometry to be symmetric about the node, which is the
    centroid of the corresponding segment.

    Parameters
    ----------
    df_segments : gpd.GeoDataFrame
        GeodataFrame with labels and polygons associated with segment labels.

    Returns
    -------
    gpd.GeoDataFrame:
        GeodataFrame with labels and LineString whose length fashions an
        upperbound of the width. Namely it has length equal to the perimeter of
        the segment it belongs, symmetric about the node that determines the
        centroid of the segment.
    """

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


def get_geo_width_df(df_segments: gpd.GeoDataFrame,
                     G: nx.Graph,
                     radius: int = 2,
                     geo_corrections: bool = True) -> gpd.GeoDataFrame:
    """
    This obtains the width geometry and related data as a GeoDataFrame. The
    geometry of the width is determined according to the node attribute in G
    `flow_vector_perp`. We then create a line centered at the node (centroid of
    the segment) whose length is equal to the perimeter of the corresponding
    segment. We make the following corrections (if geo_corrections is True):

    1. Ensures that if a width geometry contains the node, it only uses that
    the connected line segment that contains the node
    2. If the width geometry does not contain the node, then it uses the full
    width geometry determined via intersection
    3. If the geometry is a single point, return an empty linestring.

    Parameters
    ----------
    df_segments : gpd.GeoDataFrame
        The segment dataframe in which the geometry are polygons or
        multi-polygons corresponding to a segment label.
    G : nx.Graph
        The corresponding RAG graph of the segments.
    radius : int
        The radius within the RAG graph to consider intersection of the
        candidate geometry.
    geo_corrections : bool
        This permits additional geo_corrections listed above. Otherwise we
        obtain just the intersection with a segment and its neighborhood of
        specified radius.

    Returns
    -------
    gpd.GeoDataframe:
        The dataframe with the label, node, and width geometry of the
        corresponding segment.
    """
    df_candidates = get_candidate_line_df(df_segments)
    df_segments_b = buffer_segments_with_rag(G,
                                             df_segments,
                                             radius=radius)

    width_geometries = df_candidates.geometry.intersection(df_segments_b)
    df_widths = gpd.GeoDataFrame({'label': df_segments.label,
                                  'node': df_segments.node},
                                 geometry=width_geometries,
                                 crs=df_segments.crs
                                 )

    if geo_corrections:
        df_widths.geometry = df_widths.aggregate(_update_width_geometry,
                                                 axis=1)

    df_widths['width_m'] = df_widths.geometry.length

    # Let's make sure after all that we have all the nodes accounted for
    # Some of the nodes may have been pruned
    assert(set(G.nodes).issubset(set(df_widths.node)))
    return df_widths


def _update_width_geometry(row: gpd.GeoSeries) -> LineString:
    """
    Performs the basic operations of updating the geometry via:

    1. Ensures that if a width geometry contains the node, it only uses that
    the connected line segment that contains the node
    2. If the width geometry does not contain the node, then it uses the full
    width geometry determined via intersection
    3. If the geometry is a single point, return an empty linestring.

    This function is applied via df.map(_update_width_geometry), so the row is
    a geoseries with index corresponding to columns.

    Parameters
    ----------
    row : gpd.GeoSeries
        This row is from the df_widths dataframe `get_geo_width_df` with
        columns label, node, and a LineString geometry.

    Returns
    -------
    LineString:
        The updated LineString using the manipulations above.
    """
    width_geometry = row['geometry']
    node = row['node']

    width_line = width_geometry
    if isinstance(width_geometry, (MultiLineString, GeometryCollection)):
        def get_segment_with_node(line_seg):
            return line_seg.buffer(1).contains(Point(node))
        filtered_lines = list(filter(get_segment_with_node, width_line))
        # Get the first filtered line
        if filtered_lines:
            width_line = filtered_lines[0]
            if isinstance(width_line, Point):
                return LineString()
        # Empty geometry collection
        else:
            width_line = LineString()
    elif isinstance(width_geometry, Point):
        width_line = LineString()
    return width_line


##################
# Final Width
# Determination
##################

def update_nodes_with_geometric_width_data(G: nx.DiGraph,
                                           width_df: gpd.GeoDataFrame) \
                                                   -> nx.Graph:
    """
    Uses width_df (GeoDataFrame) and uses the length of line strings
    to add `width_m` to node attributes.

    Parameters
    ----------
    G : nx.DiGraph
        Channel network graph
    width_df : gpd.GeoDataFrame
        Width df with column `width_m`.

    Returns
    -------
    nx.Graph:
       Input graph modified inplace.
    """
    df_dict = width_df.set_index('node').to_dict()
    widths_dict = df_dict['width_m']

    nodes = list(G.nodes())
    node_att_updates = {node: {'width_m': widths_dict[node],
                               }
                        for node in nodes}
    nx.set_node_attributes(G, node_att_updates)
    return G


def update_edge_width_m(G: nx.Graph) -> nx.Graph:
    """
    Updates edge widths as average width from nodes that determine an edge and
    the node attribute `width_m`.

    Parameters
    ----------
    diG : nx.Graph
        Graph with node attribute `width_m`.

    Returns
    -------
    nx.Graph:
        Input graph updated in place.
    """

    # Get Node Data
    node_width_dict = nx.get_node_attributes(G, 'width_m')
    if not node_width_dict:
        raise ValueError('diG must have `width_m` attribute')

    # Get Edge Data
    edge_data_ = (G.edges(data=True))
    edge_data = {(e[0], e[1]): e[2] for e in edge_data_}
    edge_data.update({(e[1], e[0]): e[2] for e in edge_data_})
    edges = edge_data.keys()

    def get_edge_width(edge):
        width_0 = node_width_dict[edge[0]]
        width_1 = node_width_dict[edge[1]]
        return (width_0 + width_1) / 2
    edge_width_dict = {edge: {'width_m': get_edge_width(edge)}
                       for edge in edges}
    nx.set_edge_attributes(G, edge_width_dict)
    return G


def update_graph_with_widths(G: nx.Graph,
                             width_df: gpd.GeoDataFrame) -> nx.Graph:
    """
    Updates edge and node attributes using width_df, which has 'label' column
    corresponding to the segment/node of the RAG and a 'width_m' and its
    geometry corresponding to this estimated width. The edge `width_m` is
    assigned as the average `width_m` of the nodes that determine it.

    Parameters
    ----------
    G : nx.Graph
        The RAG graph of the channel network.
    width_df : gpd.GeoDataFrame
        The corresponding width_df with labels, width_m, and width_m LineString
        corresponding to the geometry along which the width is measured.

    Returns
    -------
    nx.Graph:
        The input graph modified in place.
    """
    G = update_nodes_with_geometric_width_data(G, width_df)
    G = update_edge_width_m(G)
    return G
