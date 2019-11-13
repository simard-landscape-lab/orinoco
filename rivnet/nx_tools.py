import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString, Point
import pandas as pd
import scipy.ndimage as nd
import numpy as np


def get_RAG_neighbors(label_array):
    """
    Assume 0 is mask label and ignore.
    """

    indices = nd.find_objects(label_array)
    labels_unique = np.unique(label_array)
    neighbors = {}

    for k, label in enumerate(labels_unique):

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

        dist = nd.distance_transform_edt(~label_mask_in_slice, return_distances=True, return_indices=False)
        neighbors_temp = np.unique(label_slice[(dist > 0) & (dist < 2) & ~mask_slice])
        neighbors[label] = neighbors_temp

    return neighbors


def remove_tuples_from_df(df):
    column_data = [(key, value, type(value)) for key, value in df.iloc[0].to_dict().items()]
    tuple_cols = list(filter(lambda item: item[2] == tuple, column_data))

    for item in tuple_cols:
        new_cols = [item[0] + f'_{k}' for k in range(len(item[1]))]
        # from https://stackoverflow.com/questions/29550414/how-to-split-column-of-tuples-in-pandas-dataframe
        df[new_cols] = pd.DataFrame(df[item[0]].tolist(), index=df.index)
        df.drop(item[0], axis=1, inplace=True)
    cols = [col for col in df.columns if col != 'geometry'] + ['geometry']
    df = df[cols]
    return df


def export_edges_to_geodataframe(G, directed_info=False):
    edge_data = list(G.edges(data=True))
    edge_dict = {(e[0], e[1]): e[2] for e in edge_data}
    if directed_info:
        edge_records = [{**{'head': edge[0], 'tail': edge[1]},
                         **edge_dict[edge]} for edge in edge_dict.keys()]
    else:
        edge_records = [{**edge_dict[edge]} for edge in edge_dict.keys()]

    geometry = [LineString([edge[0], edge[1]]) for edge in edge_dict.keys()]
    df_geo = gpd.GeoDataFrame(edge_records,
                              geometry=geometry, crs={'init': 'epsg:4326'})
    return df_geo


def export_nodes_to_geodataframe(G):
    node_attributes = dict(G.nodes(data=True))
    nodes = list(node_attributes.keys())
    node_data = [node_attributes[node] for node in nodes]
    geometry = [Point(node) for node in nodes]
    df_geo = gpd.GeoDataFrame(node_data, geometry=geometry, crs={'init': 'epsg:4326'})
    return df_geo


def check_edge_set_membership(edge, edge_pairs):
    check_1 = (edge[0], edge[1]) in edge_pairs
    check_2 = (edge[1], edge[0]) in edge_pairs
    membership = bool(check_1 or check_2)
    return membership


def dfs_line_search(G: nx.Graph, source):
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

                # Degree > 2
                if nx.degree(G, child) > 2:
                    new_edges = [(child, v) for v in G.neighbors(child) if v != parent]
                    stack += new_edges
                    break

                # Else resume line creation
                parent, child = child, new_children[0]
                current_edge = (parent, child)

        if line:
            yield line
