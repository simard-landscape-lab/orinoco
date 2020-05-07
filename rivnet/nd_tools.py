import numpy as np
import scipy.ndimage as nd
from functools import wraps
from scipy.ndimage import find_objects


def fill_mask_with_constant_value(band_func=None, *, fill_value=0):
    """
    source: https://stackoverflow.com/questions/3888158/making-decorators-with-optional-arguments
    """
    def fill_mask(band_func_input):
        """
        a wrapper to ensure that mask values are filled with constant, prior
        to application
        """
        @wraps(band_func_input)
        def band_func_mod(img, *args, **kwargs):
            if len(img.shape) != 2:
                raise ValueError('Img must be a 2d array')
            mask = kwargs.pop('mask', None)
            if mask is None:
                mask = np.zeros(img.shape)
            mask = mask.astype(bool)
            out_img = img.copy()
            out_img[mask] = fill_value
            out_img = band_func_input(out_img, *args, **kwargs)
            if np.any(mask):
                out_img[mask] = img[mask][0]
            return out_img
        return band_func_mod
    # occurs when no keyword arguments used (only sees decorated function)
    if band_func:
        return fill_mask(band_func)
    # occurs when keyword arguments used (sees that original_predict is None)
    else:
        return fill_mask


def get_array_from_features(label_array: np.ndarray, features: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    label_array:
        p x q Integer array of labels corresponding to superpixels
    features:
        m x n array of features - M corresponds to number of distinct items to be classified and N number of features for each item.

    Returns
    -------
    out:
        p x q (x n) array where we drop the dimension if n == 1.

    Notes
    ------

    From features and labels, obtain an array with each label populated with correct measurement.

    Inverse of get_features_from_array with fixed labels, namely if `f` are features and `l` labels, then

        get_features_from_array(l, get_array_from_features(l, f)) == f
    """
    # Assume labels are 0, 1, 2, ..., n
    if len(features.shape) != 2:
        raise ValueError('features must be 2d array')
    elif features.shape[1] == 1:
        out = np.zeros(label_array.shape)
    else:
        m, n = label_array.shape
        out = np.zeros((m, n, features.shape[1]))

    labels_p1 = label_array + 1
    indices = find_objects(labels_p1)
    labels_unique = np.unique(labels_p1)
    # determine that (number of features) == (number of unique superpixel labels)
    assert(len(labels_unique) == features.shape[0])
    for k, label in enumerate(labels_unique):
        indices_temp = indices[label-1]
        # if features is m x 1, then do not need extra dimension when indexing
        if features.shape[1] == 1:
            out[indices_temp][labels_p1[indices_temp] == label] = features[k, 0]
        # if features is m x n with n > 1, then requires extra dimension when indexing
        else:
            out[indices_temp + (np.s_[:], )][labels_p1[indices_temp] == label] = features[k, ...]
    return out


def get_features_from_array(label_array: np.ndarray, data_array: np.ndarray) -> np.ndarray:
    """
    From single image and labels, obtain features with appropriate shape.

    Inverse of get_features_from_array with fixed labels, namely if `f` are features and `l` labels, then

        get_features_from_array(l, get_array_from_features(l, f)) == f
    """
    # Ensure that 2d label_array has the same 1st two dimensions as data_array
    assert(label_array.shape == (data_array.shape[0], data_array.shape[1]))
    labels_p1 = label_array + 1
    indices = find_objects(labels_p1)
    labels_unique = np.unique(labels_p1)

    m = len(labels_unique)
    if len(data_array.shape) == 2:
        features = np.zeros((m, 1))
    elif len(data_array.shape) == 3:
        features = np.zeros((m, data_array.shape[2])).astype(bool)
    else:
        raise ValueError('data_array must be 2d or 3d')

    for k, label in enumerate(labels_unique):
        indices_temp = indices[label-1]
        # if features is m x 1, then do not need extra dimension when indexing
        if features.shape[1] == 1:
            # import pdb; pdb.set_trace()
            features[k, 0] = data_array[indices_temp][labels_p1[indices_temp] == label][0]
        # if features is m x n with n > 1, then requires extra dimension when indexing
        else:
            features[k, ...] = data_array[indices_temp + (np.s_[:], )][labels_p1[indices_temp] == label][0, ...]
    return features


def apply_func_to_superpixels(func,
                              labels: np.ndarray,
                              array: np.ndarray) -> np.ndarray:
    """
    This is a wrapper for `scipy.ndimage.labeled_comprehension`.

    It applies the func to the array of pixels within each label and returns the measurements.
    """
    if len(array.shape) != 2:
        raise ValueError('The array must be a 2d array')
    labels_ = labels + 1
    labels_unique = np.unique(labels_)
    new_change_measurements = nd.labeled_comprehension(array, labels_, labels_unique, func, float, 0)
    return new_change_measurements.reshape((-1, 1))


def get_superpixel_area_as_features(labels):
    return apply_func_to_superpixels(np.size, labels, labels).astype(int)


@fill_mask_with_constant_value(fill_value=0)
def filter_binary_array_by_min_size(binary_array: np.ndarray,
                                    min_size: int,
                                    structure: np.ndarray = np.ones((3, 3)),
                                    mask: np.ndarray = None) -> np.ndarray:

    binary_array_temp = binary_array[~np.isnan(binary_array)]
    if ~((binary_array_temp == 0) | (binary_array_temp == 1)).all():
        raise ValueError('Array must be binary!')
    connected_component_labels, _ = nd.measurements.label(binary_array, structure=structure)
    size_features = get_superpixel_area_as_features(connected_component_labels)
    binary_features = get_features_from_array(connected_component_labels, binary_array)

    # Only want 1s of certain size
    filtered_size_features = (size_features >= min_size).astype(int) * binary_features

    binary_array_filtered = get_array_from_features(connected_component_labels, filtered_size_features)
    return binary_array_filtered


def scale_img(img: np.ndarray,
              new_min: int = 0,
              new_max: int = 1) -> np.ndarray:
    """
    Scale an image by the absolute max and min in the array to have dynamic range new_min to new_max.
    """
    i_min = np.nanmin(img)
    i_max = np.nanmax(img)
    if i_min == i_max:
        # then image is constant image and clip between new_min and new_max
        return np.clip(img, new_min, new_max)
    img_scaled = (img - i_min) / (i_max - i_min) * (new_max - new_min) + new_min
    return img_scaled


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
    + this is a rough approximation - the max in a segment is not quite right and using the distance transform does not
      consider flow direction, centroid position, channels lack of symmetry, etc.
    """
    transform = profile['transform']
    if transform.a != - transform.e:
        raise ValueError('Unequal x/y resolutions in channel mask and cannot use scipy')
    resolution = transform.a

    channel_mask = (label_array != 0).astype(np.uint8)

    channel_dist = nd.distance_transform_edt(channel_mask) * resolution
    # Because scipy determines distance using pixel's centers we overcount the 1/2 boundary pixels not in the channel
    # This could be an *underestimate* as this could be diagonal which would be approximately \sqrt(2) / 2
    width_features = 2 * (apply_func_to_superpixels(np.nanmax, label_array, channel_dist) - .5 * resolution)
    width_features[0] = np.nan
    return width_features
