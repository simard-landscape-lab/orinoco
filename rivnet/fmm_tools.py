from skimage import measure
import numpy.ma as ma
import skfmm
import numpy as np
from .nd_tools import filter_binary_array_by_min_size


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


def get_distance_segments(distance: np.array, pixel_step: int, dx: float, dy: float):
    threshold = min(dx, dy) * pixel_step
    dist_threshold = (distance / threshold)
    dist_temp = dist_threshold + 1
    dist_temp[np.isnan(dist_threshold)] = 0
    dist_temp = dist_temp.astype(np.int32)
    labels = measure.label(dist_temp, background=0, neighbors=8, connectivity=8).astype(np.int32)
    return labels
