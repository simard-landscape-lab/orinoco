import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString, Point
import pandas as pd
import scipy.ndimage as nd
import numpy as np
from typing import Callable
from tqdm import tqdm


def get_graph_from_edge_dataframe(edge_df, directed=True):

    G = nx.DiGraph() if directed else nx.Graph()

    geometry = edge_df.geometry
    head = geometry.map(lambda geo: geo.coords[0])
    tail = geometry.map(lambda geo: geo.coords[1])

    edge_df['head'] = head.values
    edge_df['tail'] = tail.values

    edge_data = edge_df.set_index(['head', 'tail']).to_dict('index')

    G.add_edges_from(edge_data.keys())
    nx.set_edge_attributes(G, edge_data)

    return G


def get_RAG_neighbors(label_array, label_subset: list = None, connectivity=4):
    """
    Assume 0 is mask label and ignore.

    Label subset is to obtain neighbors for only a subset of labels in label_array.

    Connectivity determines how the buffer is determined either with 4 or 8 connectivity.
    """

    indices = nd.find_objects(label_array)
    labels_unique = np.unique(label_array) if label_subset is None else label_subset
    neighbors = {}

    for k, label in tqdm(enumerate(labels_unique), total=len(labels_unique), desc='rag neighbors'):

        if label == 0:
            continue

        indices_temp = indices[label-1]
        sy, sx = indices_temp
        sy = np.s_[max(sy.start - 1, 0): sy.stop + 1]
        sx = np.s_[max(sx.start - 1, 0): sx.stop + 1]
        indices_temp = sy, sx

        label_mask_in_slice = (label_array[indices_temp] == label)
        label_slice = label_array[indices_temp]
        mask_slice = (label_slice == 0)

        structure_dict = {4: None,
                          8: np.ones((3, 3))}
        structure = structure_dict[connectivity]
        dilated = nd.morphology.binary_dilation(label_mask_in_slice,
                                                iterations=1,
                                                structure=structure).astype(bool)
        neighbors_temp = np.unique(label_slice[(~label_mask_in_slice) & (dilated) & ~mask_slice])
        neighbors[label] = neighbors_temp

    return neighbors


def change_tuples_to_vector(df, columns_to_ignore=None):
    first_index = list(df.index)[0]
    column_data = [(column_name, column_value, type(column_value))
                   for column_name, column_value in df.iloc[first_index].to_dict().items()]
    tuple_cols = list(filter(lambda item: item[2] == tuple, column_data))
    if columns_to_ignore:
        tuple_cols = list(filter(lambda item: item[0] not in columns_to_ignore, column_data))

    df_new = df.copy()
    for (column_name, column_value, _) in tuple_cols:
        new_cols = [column_name + f'_{coord_label}' for coord_label in ['x', 'y']]
        # from https://stackoverflow.com/questions/29550414/how-to-split-column-of-tuples-in-pandas-dataframe
        df_new[new_cols] = pd.DataFrame(df_new[column_name].tolist(), index=df.index)
        df_new.drop(column_name, axis=1, inplace=True)
    cols = [col for col in df_new.columns if col != 'geometry'] + ['geometry']
    df_new = df_new[cols]
    return df_new


def export_edges_to_geodataframe(G: nx.classes.graph.Graph, crs: dict, directed_info=False):
    edge_data = list(G.edges(data=True))
    edge_dict = {(e[0], e[1]): e[2] for e in edge_data}
    if directed_info:
        edge_records = [{**{'head': list(edge[0]), 'tail': list(edge[1])},
                         **edge_dict[edge]} for edge in edge_dict.keys()]
    else:
        edge_records = [{**edge_dict[edge]} for edge in edge_dict.keys()]

    geometry = [LineString([edge[0], edge[1]]) for edge in edge_dict.keys()]
    df_geo = gpd.GeoDataFrame(edge_records,
                              geometry=geometry, crs=crs)
    return df_geo


def export_nodes_to_geodataframe(G: nx.classes.graph.Graph, crs: dict):
    node_attributes = dict(G.nodes(data=True))
    nodes = list(node_attributes.keys())
    node_data = [node_attributes[node] for node in nodes]
    geometry = [Point(node) for node in nodes]
    df_geo = gpd.GeoDataFrame(node_data, geometry=geometry, crs=crs)
    return df_geo


def dfs_line_search(G: nx.classes.graph.Graph,
                    source: tuple,
                    break_func: Callable = None):
    """Meant for a connected graph G to yeild edges in the form [edge_0, edge_1, ..., edge_n]
    so that edge_0[0] is a (juncture, source, or sink) and edge_n[1] is a (juncture, source, or sink)
    and all nodes in between have degree 2, that is unique parent and child.

    Derived from traversal algorithms from networkx.
    """
    stack = [(source, v) for v in G.neighbors(source)]
    discovered = []

    while stack:
        current_edge = stack.pop(0)
        parent, child = current_edge[0], current_edge[1]
        line = []

        while True:
            if set(current_edge) in discovered:
                break
            else:
                line.append(current_edge)
                discovered.append(set(current_edge))
                new_children = [v for v in G.neighbors(child) if v != parent]

                # Degree 1
                if not new_children:
                    break

                # Undirected case
                # Degree > 2
                if not G.is_directed() and (G.degree(child) > 2):
                    new_edges = [(child, v) for v in G.neighbors(child) if v != parent]
                    stack += new_edges
                    break

                # Directed case; check in degree and out degree separately
                if G.is_directed() and ((G.out_degree(child) > 1) or
                                        (G.in_degree(child) > 1)):
                    new_edges = [(child, v) for v in G.neighbors(child) if v != parent]
                    stack += new_edges
                    break

                if break_func is not None and break_func(child):
                    new_edges = [(child, v) for v in G.neighbors(child) if v != parent]
                    stack += new_edges
                    break

                # Else resume line creation
                parent, child = child, new_children[0]
                current_edge = (parent, child)

        if line:
            yield line
