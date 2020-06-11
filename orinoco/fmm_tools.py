from skimage import measure
import numpy.ma as ma
import skfmm
import numpy as np
from .nd_tools import (filter_binary_array_by_min_size,
                       get_superpixel_area_as_features,
                       apply_func_to_superpixels,
                       get_array_from_features)

from .nx_tools import get_RAG_neighbors
from skimage.segmentation import relabel_sequential


def get_distance_along_channel_using_fmm(water_mask: np.array,
                                         init_mask: np.array,
                                         dx: float,
                                         dy: float,
                                         area_threshold: float = 0.) -> np.array:
    """
    Obtain distance array from binary mask and initialization mask (the latter representing the ocean).
    Requires specification of resolution, but can trivially set dx=dy=1 for experimentation.

    Parameters
    ----------
    water_mask : np.array
        Binary water mask (will be recast to boolean)
    init_mask : np.array
        Ocean mask or the region where we want our flow to originate
    dx : float
        Horizontal resolution
    dy : float
        Vertical Resolution
    area_threshold : float
        Remove connected areas whose pixel size whose relative size is
        below this threshold. Defaults to 0.

    Returns
    -------
    np.array:
       Distance array, same dimensions as water_mask with nodata areas having value np.nan
    """

    mask = ~(water_mask.astype(bool))
    phi = ma.masked_array(np.ones(water_mask.shape), mask=mask)
    phi.data[init_mask.astype(bool)] = 0

    dist = skfmm.distance(phi, dx=(dy, dx))

    # Adjust Distance Array
    dist_data = dist.data.astype(np.float32)
    # remove areas close to the ocean interface
    dist_mask = mask
    # We set the ocean to np.nan
    dist_mask = dist_mask | (dist_data <= 0.)
    dist_data[dist_mask] = np.nan

    # Remove Areas that are smaller than 2.5 percent of total area
    # The ocean mask and the water mask may not agree near the interface and this removes
    # Areas of erroneous additions
    if area_threshold > 0.:
        binary_array = (~np.isnan(dist_data)).astype(np.uint8)
        min_size = np.sum(binary_array > 0) * area_threshold
        size_mask = filter_binary_array_by_min_size(binary_array, min_size).astype(bool)
        dist_data[~size_mask] = np.nan

    return dist_data


def merge_labels_below_minsize(labels: np.array, min_size: int, connectivity: int = 4) -> np.array:
    """
    Takes labels below min_size and merges a label with a connected neighbor (with respect to the connectivity description).

    Parameters
    ----------
    labels : np.array
        2d label array.
    min_size : int
        Keeps only segments of at least this size
    connectivity : int
        4 or 8 connectivity accepted.

        See: https://en.wikipedia.org/wiki/Pixel_connectivity

    Returns
    -------
    np.array:
        Updated 2d label array
    """
    size_features = get_superpixel_area_as_features(labels)
    unique_labels = np.arange(0, labels.max() + 1)
    labels_to_merge = unique_labels[size_features.ravel() < min_size]
    neighbor_dict = get_RAG_neighbors(labels, label_subset=labels_to_merge, connectivity=connectivity)

    def merger(label_arr):
        label = label_arr[0]
        neighbors = neighbor_dict.get(label)
        if neighbors is not None and len(neighbors) > 0:
            return neighbors[0]
        else:
            return label
    label_features = apply_func_to_superpixels(merger, labels, labels, dtype=int)
    labels = get_array_from_features(labels, label_features)
    labels, _, _ = relabel_sequential(labels)
    return labels


def get_distance_segments(distance: np.array,
                          n_pixel_threshold: int,
                          dx: float,
                          dy: float,
                          connectivity: int = 4,
                          min_size: int = 4) -> tuple:
    """


    Parameters
    ----------
    distance : np.array
        Distance array determined using fmm
    n_pixel_threshold : int
        Segments are determined according to int(distance / threshold), ensuring connectivity, where
        threshold = min(dx, dy) * n_pixel_threshold.
    dx : float
        Horizontal resolution
    dy : float
        Vertical resolution
    connectivity : int
        4 or 8 connectivity accepted.
        See: https://en.wikipedia.org/wiki/Pixel_connectivity
    min_size : int
       Minimum Segment size. Default 4.

    Returns
    -------
    tuple:
       labels, interface_adjacent_labels

        + labels is a 2d array with the segmentation in which each pixel corresponds to a segment label.
        + interface_adjacent_labels is an array of labels adjacent to the interface determined via dist,
        namely those segments less than threshold
    """
    threshold = min(dx, dy) * n_pixel_threshold
    dist_threshold = (distance / threshold)
    # We ensure that our background is 0
    dist_temp = dist_threshold + 1
    # We assume that the 0 label is background
    dist_temp[np.isnan(dist_threshold)] = 0
    dist_temp = dist_temp.astype(np.int32)

    # 4-connectivty is 1-hop connectivity for skimage.measure.label and similarly 8-connectivity is
    # called 2-hop connectivity.
    connectivity_label = 2 if connectivity == 8 else 1
    labels = measure.label(dist_temp, background=0, connectivity=connectivity_label).astype(np.int32)

    if min_size is not None:
        labels = merge_labels_below_minsize(labels, min_size, connectivity=connectivity)

    # Obtain labels adjacent to interface
    # Look at those after thresholding are within 1 (must look at min after merging)
    min_distance_features = apply_func_to_superpixels(np.min, labels, dist_threshold).ravel()

    # Weird indexing below to avoid warning due to np.nan which occurs at index 0.
    # Namely, we ignore index 1 by using [1:] and since argwhere assumes input is properly formed
    # we must add 1 back.
    # Note our features are such that each index corresponds to that label!
    interface_adjacent_labels = np.argwhere(min_distance_features[1:] < 1) + 1
    return labels, interface_adjacent_labels
