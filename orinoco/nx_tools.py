import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point
import pandas as pd
import scipy.ndimage as nd
import numpy as np
from typing import Callable, Generator, Dict
from tqdm import tqdm


def get_graph_from_edge_dataframe(edge_df: gpd.GeoDataFrame,
                                  directed: bool = True) -> nx.Graph:
    """
    Obtain a networkX Graph or Digraph from an edge dataframe as exported from
    'export_edges_to_geodataframe'.

    Parameters
    ----------
    edge_df : pd.DataFrame
        Edges encoded as shapely.geometry.LineString. If directed, assume
        edge[0] --> edge[1] indicates direction of edge from parent to child.
    directed : bool
        Whether to use undirected or directed Graph.

    Returns
    -------
    nx.Graph:
        The graph with edges and edge attributes from geodataframe.
    """

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


def get_RAG_neighbors(label_array: np.ndarray,
                      label_subset: list = None,
                      connectivity: int = 4) -> Dict[int, list]:
    """
    Obtains dictionary of lists. Given a label i, the value of key i is a list
    of the neighbors with respect to the RAG.

    Require 0 to be mask label and neither include in dictionary nor as
    neighbors for any nodes.

    Label subset is to obtain neighbors for only a subset of labels in
    label_array. Connectivity determines how the RAG is computed on the
    original label array. Accepted are 4 or 8 connectivity.

    Parameters
    ----------
    label_array : np.ndarray
        p x q array of labels
    label_subset : list
        subset of labels. Default is None in which all labels are used
    connectivity : int
        4- or 8- connectivity to be used. Default is 4.

    Returns
    -------
    Dict[int, list]:
       Dict[i] = [neighbor1, neighbor2, ...]
    """
    indices = nd.find_objects(label_array)
    if label_subset is None:
        labels_unique = np.unique(label_array)
    else:
        labels_unique = list(set(label_subset))
    neighbors = {}

    for k, label in tqdm(enumerate(labels_unique),
                         total=len(labels_unique),
                         desc='rag neighbors'):

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
                                                structure=structure)
        dilated = dilated.astype(bool)
        neighbors_temp = np.unique(label_slice[(~label_mask_in_slice)
                                               & (dilated) & ~mask_slice])
        neighbors[label] = neighbors_temp

    return neighbors


def split_tuple_pairs(df: pd.DataFrame,
                      cols_to_ignore: list = None) -> pd.DataFrame:
    """
    Assuming dataframe df has a column with values `(x, y)`, then splits column
    with `column_name` into two columns with names `column_name_x`,
    `column_name_y` with values `x` and `y` respectively.

    This is to allow for geodataframes to be saved to geojson files in which
    tuples and lists are not permitted as attribute values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe whose columns with 2-tuples to be split into separate
        columns.
    cols_to_ignore : list
        Removes these columns from the new dataframe

    Returns
    -------
    pd.DataFrame:
        A dataframe with whose 2-tuple columns are split into separate columns.
    """
    first_index = list(df.index)[0]
    first_row_data = df.iloc[first_index].to_dict().items()
    column_data = [(col_name, col_value, type(col_value))
                   for (col_name, col_value) in first_row_data]
    tuple_cols = list(filter(lambda item: item[2] == tuple, column_data))
    if cols_to_ignore:
        tuple_cols = list(filter(lambda item: item[0] not in cols_to_ignore,
                                 column_data))

    df_new = df.copy()
    for (column_name, column_value, _) in tuple_cols:
        new_cols = [column_name + f'_{coord_label}'
                    for coord_label in ['x', 'y']]
        # from
        # https://stackoverflow.com/questions/29550414/how-to-split-column-of-tuples-in-pandas-dataframe
        df_new[new_cols] = pd.DataFrame(df_new[column_name].tolist(),
                                        index=df.index)
        df_new.drop(column_name, axis=1, inplace=True)
    cols = [col for col in df_new.columns if col != 'geometry'] + ['geometry']
    df_new = df_new[cols]
    return df_new


def export_edges_to_geodataframe(G: nx.classes.graph.Graph,
                                 crs: dict) -> gpd.GeoDataFrame:
    """
    Take a graph and a coordinate reference system and export the graph edges
    and their attributes to a geodataframe for plotting and saving to common
    GIS formats.

    We assume each node is given as (x, y) in R^2 associated with the
    appropriate CRS.

    Parameters
    ----------
    G : nx.classes.graph.Graph
        Graph to export (can be directed or undirected).
    crs : dict
        Using the rasterio convention: `{'init': 'epsg:<epsg_code>'}`.

    Returns
    -------
    gpd.GeoDataFrame:
        The geodataframe with edges as LineStrings (see shapely.geometry) and
        all edge attributes.
    """
    edge_data = list(G.edges(data=True))
    edge_dict = {(e[0], e[1]): e[2] for e in edge_data}
    edge_records = [{**edge_dict[edge]} for edge in edge_dict.keys()]

    geometry = [LineString([edge[0], edge[1]]) for edge in edge_dict.keys()]
    df_geo = gpd.GeoDataFrame(edge_records,
                              geometry=geometry, crs=crs)
    return df_geo


def export_nodes_to_geodataframe(G: nx.classes.graph.Graph,
                                 crs: dict) -> gpd.GeoDataFrame:
    """
    Take a graph and a coordinate reference system and export the graph nodes
    and their attributes to a geodataframe of Shapely Ponits for plotting and
    saving to common GIS formats.

    Parameters
    ----------
    G : nx.classes.graph.Graph
        Graph to export
    crs : dict
        Using the rasterio convention: `{'init': 'epsg:<epsg_code>'}`.

    Returns
    -------
    gpd.GeoDataFrame:
       Geodataframe with nodes as Points (see shapely.geometry) and all node
       attributes.
    """
    node_attributes = dict(G.nodes(data=True))
    nodes = list(node_attributes.keys())
    node_data = [node_attributes[node] for node in nodes]
    geometry = [Point(node) for node in nodes]
    df_geo = gpd.GeoDataFrame(node_data, geometry=geometry, crs=crs)
    return df_geo


def dfs_line_search(G: nx.classes.graph.Graph,
                    source: tuple,
                    break_func: Callable = None) -> Generator[list,
                                                              None,
                                                              None]:
    """
    Meant for a connected graph G to yeild edges in the form [edge_0, edge_1,
    ..., edge_n] so that edge_0[0] is a (juncture or degree 1) and edge_n[1] is
    a (juncture or degree 1) and all nodes in between have degree 2.

    Derived from traversal algorithms from networkx.

    Parameters
    ----------
    G : nx.classes.graph.Graph
        Graph to be partitioned
    source : tuple
        Starting node to partition edges. Must ensure that source is valid
        divider (junction or degree 1) otherwise single edge group will be
        divided into separate groups.
    break_func : Callable
        Additional function to break. In this case, we use the `interface_adj`
        of our channel network to break from a junction because sometimes such
        a node will not have degree 1.

    Returns
    -------
    Generator[list, None, None]:
        List of edges within one edge grouping
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
                    new_edges = [(child, v)
                                 for v in G.neighbors(child) if v != parent]
                    stack += new_edges
                    break

                # Directed case; check in degree and out degree separately
                if G.is_directed() and ((G.out_degree(child) > 1) or
                                        (G.in_degree(child) > 1)):
                    new_edges = [(child, v)
                                 for v in G.neighbors(child) if v != parent]
                    stack += new_edges
                    break

                if break_func is not None and break_func(child):
                    new_edges = [(child, v)
                                 for v in G.neighbors(child) if v != parent]
                    stack += new_edges
                    break

                # Else resume line creation
                parent, child = child, new_children[0]
                current_edge = (parent, child)

        if line:
            yield line
