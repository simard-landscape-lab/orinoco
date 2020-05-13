import numpy as np
import networkx as nx
from tqdm import tqdm
from shapely.geometry import (LineString,
                              Point,
                              GeometryCollection,
                              MultiLineString,
                              )
import geopandas as gpd
import concurrent.futures
from shapely.errors import TopologicalError


def unit_vector(vector: np.ndarray):
    return vector / np.linalg.norm(vector)


def angle_between(v1: np.ndarray, v2: np.ndarray):
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


def get_flow_vector(node: tuple, river_graph: nx.Graph,
                    weighted_by_distance: bool = True,
                    inv_dist_power: int = 2) -> np.ndarray:
    """
    Takes node and river_graph and obtain direction of flow at node dictated by network structure.

    Take a weighted average of neighboring vectors to determine which direction is most likely at
    given node. Weighted according to (1/distance) ** inv_dist_power
    """
    edge_neighbors = [(node, neighbor) for neighbor in nx.all_neighbors(river_graph, node)]
    neighbor_vectors = list(map(edge_to_vector, edge_neighbors))

    def realign_vectors_partial(edge_vector):
        return realign_vector(edge_vector, reference_edge=neighbor_vectors[0])
    neighbor_vectors = list(map(realign_vectors_partial, neighbor_vectors))

    neighbor_vectors_stacked = np.stack(neighbor_vectors, axis=0)

    norms = np.linalg.norm(neighbor_vectors_stacked, axis=1).reshape((neighbor_vectors_stacked.shape[0], -1))
    neighbor_unit_vectors_stacked = neighbor_vectors_stacked / norms

    if weighted_by_distance:
        weights = 1. / ((norms)**inv_dist_power)
        weights /= np.sum(weights)
        est_grad = np.sum(neighbor_unit_vectors_stacked * weights, axis=0)
    else:
        est_grad = np.mean(neighbor_unit_vectors_stacked, axis=0)
    return est_grad


def add_flow_attributes(G: nx.Graph):
    nodes = list(G.nodes())

    def get_flow_vector_partial(node):
        return get_flow_vector(node, G, weighted_by_distance=True)
    grad_vectors = list(map(get_flow_vector_partial, tqdm(nodes, desc='flow vector computation')))
    grad_vectors_perp = list(map(_perp, grad_vectors))
    node_indices = list(range(len(nodes)))
    node_att_updates = {nodes[k]: {'flow_vector': tuple(grad_vectors[k]),
                                   'flow_vector_perp': grad_vectors_perp[k]} for k in node_indices}
    nx.set_node_attributes(G, node_att_updates)
    return G


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


def get_width_geometry_for_node(node: tuple, transform,
                                water_geoseries: gpd.GeoSeries,
                                flow_perp_dict: dict,
                                width_search_buffer: float = 500):
    """
    There may be a faster way to do this, but currently just creates a really long line and intersects with the water geometry
    """

    direction = flow_perp_dict[node]

    length = transform[0] * width_search_buffer
    p0 = get_vector_tail(node, np.array(direction) * 1, length)
    p1 = get_vector_tail(node, np.array(direction) * -1, length)

    width_line_temp = LineString([p0, p1])

    intersection_geo_series = water_geoseries.intersection(width_line_temp)
    nonempty_intersections = intersection_geo_series.is_empty
    try:
        width_line = intersection_geo_series[~nonempty_intersections].unary_union
    except TopologicalError:
        print(f'Topology Exception occured for Node {node} and width_line_temp {width_line_temp.coords[:]}')
        width_line = LineString()

    # Assume transform[0] = transform[4]
    delta = transform[0]

    if isinstance(width_line, (MultiLineString, GeometryCollection)):

        def get_segment_with_node(line_seg):
            return line_seg.buffer(delta).contains(Point(node))
        filtered_lines = list(filter(get_segment_with_node, width_line))
        if filtered_lines:
            width_line = filtered_lines[0]
        else:
            width_line = LineString()
    if isinstance(width_line, Point):
        width_line = LineString()
    return width_line


def get_width_geometries(rivG, water_mask_df, profile, multithreaded=True, max_workers=25):
    flow_perp_dict = nx.get_node_attributes(rivG, 'flow_vector_perp')
    nodes = list(rivG.nodes())

    water_geometry = water_mask_df.geometry
    # used buffer per this suggestions with multipolygons: https://github.com/gboeing/osmnx/issues/278
    water_geoseries = water_geometry.buffer(0)

    transform = profile['transform']

    def get_geometry_for_node(node):
        return get_width_geometry_for_node(node, transform, water_geoseries, flow_perp_dict)

    if multithreaded:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            line_geometries = list(tqdm(executor.map(get_geometry_for_node, nodes), total=len(nodes)))
    else:
        line_geometries = list(map(get_geometry_for_node, tqdm(nodes)))

    node_geometry_dict = dict(zip(nodes, line_geometries))
    return node_geometry_dict
