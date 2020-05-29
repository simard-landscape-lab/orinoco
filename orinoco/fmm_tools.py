from skimage import measure
import numpy.ma as ma
import skfmm
import numpy as np
from .nd_tools import (filter_binary_array_by_min_size,
                       get_features_from_array,
                       get_superpixel_area_as_features,
                       apply_func_to_superpixels,
                       get_array_from_features)

from .nx_tools import get_RAG_neighbors
from skimage.segmentation import relabel_sequential


def get_distance_along_river_using_fmm(water_mask: np.array,
                                       init_mask: np.array,
                                       dx: float,
                                       dy: float,
                                       area_threshold: float = .05,
                                       minimum_distance_from_init_mask: float = 1):
    mask = ~(water_mask.astype(bool))
    phi = ma.masked_array(np.ones(water_mask.shape), mask=mask)
    phi.data[init_mask.astype(bool)] = 0

    dist = skfmm.distance(phi, dx=(dy, dx))

    # Adjust Distance Array
    dist_data = dist.data.astype(np.float32)
    # remove areas close to the ocean interface
    dist_mask = mask | (dist_data <= minimum_distance_from_init_mask)
    dist_data[dist_mask] = np.nan

    # Remove Areas that are smaller than 2.5 percent of total area
    # The ocean mask and the water mask may not agree near the interface and this removes
    # Areas of erroneous additions
    if area_threshold > 0:
        binary_array = (~np.isnan(dist_data)).astype(np.uint8)
        min_size = np.sum(binary_array > 0) * area_threshold
        size_mask = filter_binary_array_by_min_size(binary_array, min_size, mask=dist_mask).astype(bool)
        dist_data[~size_mask] = np.nan

    return dist_data


def merge_labels_below_minsize(labels, min_size, connectivity=4):
    """
    Takes labels and merges a label with a connected neighbor (with respect to the connectivity description).
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
    label_features = apply_func_to_superpixels(merger, labels, labels)
    labels = get_array_from_features(labels, label_features)
    labels, _, _ = relabel_sequential(labels)
    return labels


def get_distance_segments(distance: np.array,
                          pixel_step: int,
                          dx: float,
                          dy: float,
                          connectivity: int = 4,
                          min_size: int = 4):
    threshold = min(dx, dy) * pixel_step
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
    distance_pseudo_features = get_features_from_array(labels, dist_threshold).ravel()
    # weird indexing to avoid warning due to np.nan which occurs at 0.
    interface_adjacent_labels = np.argwhere(distance_pseudo_features[1:] <= 1) + 1
    return labels, interface_adjacent_labels
