import numpy as np
import scipy.ndimage as nd
from scipy.ndimage import find_objects
from typing import Callable


def get_array_from_features(label_array: np.ndarray,
                            features: np.ndarray) -> np.ndarray:
    """
    Using p x q segmentation labels (2d) and feature array with dimension (m x
    n) where m is the number of unique labels and n is the number of features,
    obtain a p x q x m channel array in which each spatial segment is labeled
    according to n-features.

    See `find_objects` found
    [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.find_objects.html)
    for the crucial scipy function used.

    Parameters
    ----------
    label_array : np.array
        p x q integer array of labels corresponding to superpixels
    features : np.array
        m x n array of features - M corresponds to number of distinct items to
        be classified and N number of features for each item.

    Returns
    -------
    out : np.array
        p x q (x n) array where we drop the dimension if n == 1.

    Notes
    ------
    Inverse of get_features_from_array with fixed labels, namely if `f` are
    features and `l` labels, then:

        get_features_from_array(l, get_array_from_features(l, f)) == f

    And similarly, if `f_array` is an array of populated segments, then

        get_array_from_features(l, get_features_from_array(l, f)) == f
    """
    # Assume labels are 0, 1, 2, ..., n
    if len(features.shape) != 2:
        raise ValueError('features must be 2d array')
    elif features.shape[1] == 1:
        out = np.zeros(label_array.shape, dtype=features.dtype)
    else:
        m, n = label_array.shape
        out = np.zeros((m, n, features.shape[1]), dtype=features.dtype)

    labels_p1 = label_array + 1
    indices = find_objects(labels_p1)
    labels_unique = np.unique(labels_p1)
    # ensures that (number of features) == (number of unique superpixel labels)
    assert(len(labels_unique) == features.shape[0])
    for k, label in enumerate(labels_unique):
        indices_temp = indices[label-1]
        # if features is m x 1, then do not need extra dimension when indexing
        label_slice = labels_p1[indices_temp] == label
        if features.shape[1] == 1:
            out[indices_temp][label_slice] = features[k, 0]
        # if features is m x n with n > 1, then requires extra dimension when
        # indexing
        else:
            out[indices_temp + (np.s_[:], )][label_slice] = features[k, ...]
    return out


def get_features_from_array(label_array: np.ndarray,
                            data_array: np.ndarray) -> np.ndarray:
    """
    Assuming that each segment area from `label_array` (p x q) has a
    homogeneous value, obtain the corresonding feautre vector of size (m x n),
    where m is the number of segment labels and n is the number of channels in
    `data_array` (p x q x n). We also allow data array to be (p x q) if there
    is only one channel.


    See `find_objects` found
    [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.find_objects.html)
    for the crucial scipy function used.

    Parameters
    ----------
    label_array : np.ndarray
        p x q integer array of labels corresponding to superpixels
    data_array : np.ndarray
        p x q x n (or p x q) array of data assumed that each segment label has
        the same value.

    Returns
    -------
    np.ndarray:
       m x n array where m is `len(np.unique(label_array))` and n is the number
       of channels.  If `data_array` has shape p x q, then n = 1.

    Notes
    ------
    Inverse of get_features_from_array with fixed labels, namely if `f` are
    features and `l` labels, then

        get_features_from_array(l, get_array_from_features(l, f)) == f

    And similarly, if `f_array` is an array of populated segments, then

        get_array_from_features(l, get_features_from_array(l, f)) == f
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
        label_slice = labels_p1[indices_temp] == label
        if features.shape[1] == 1:
            features[k, 0] = data_array[indices_temp][label_slice][0]
        # if features is m x n with n > 1, then requires extra dimension when
        # indexing
        else:
            temp = data_array[indices_temp + (np.s_[:], )]
            features[k, ...] = temp[label_slice][0, ...]
    return features


def apply_func_to_superpixels(func: Callable,
                              labels: np.ndarray,
                              array: np.ndarray,
                              dtype: type = float) -> np.ndarray:
    """
    This is a wrapper for `scipy.ndimage.labeled_comprehension`.

    See this
    [link](https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.labeled_comprehension.html).

    Parameters
    ----------
    func : Callable
        Function to call on each flattened segment
    labels : np.ndarray
        p x q label array
    array : np.ndarray
        p x q data array
    dtype : type
        The return type of the array of features. Defaults to float.

    Returns
    -------
    np.ndarray:
        A populated (float) array in which each segment i is filled with value
        func(array[array = i]).
    """
    if len(array.shape) != 2:
        raise ValueError('The array must be a 2d array')
    labels_ = labels + 1
    labels_unique = np.unique(labels_)
    features = nd.labeled_comprehension(array, labels_, labels_unique, func, dtype, np.nan)
    return features.reshape((-1, 1))


def get_superpixel_area_as_features(labels: np.array) -> np.array:
    """
    Obtain a feature array in which features are size of corresponding features.

    Parameters
    ----------
    labels : np.array
        Label array (p x q)

    Returns
    -------
    np.array:
        Size features (m x 1), where m is number of unique labels.
    """
    return apply_func_to_superpixels(np.size, labels, labels).astype(int)


def filter_binary_array_by_min_size(binary_array: np.ndarray,
                                    min_size: int,
                                    structure: np.ndarray = np.ones((3, 3)),
                                    ) -> np.ndarray:
    """
    Look at contigious areas of 1's and if size is less than min_size, remove it.

    Parameters
    ----------
    binary_array : np.ndarray
        Array of 0's and 1's.
    min_size : int
        Minimum size
    structure : np.ndarray
        How connectivity is determined. 4-connectivity is np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).
        8-connectivity is np.ones((3, 3)), which is the default.

    Returns
    -------
    np.ndarray:
       binary array in which small continguous areas fo size less than min_size have been removed.
    """

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

    Useful for visualization.

    Parameters
    ----------
    img : np.ndarray
    new_min : int
    new_max : int

    Returns
    -------
    np.ndarray:
       New image with shape equal to img, scaled to [new_min, new_max]
    """
    i_min = np.nanmin(img)
    i_max = np.nanmax(img)
    if i_min == i_max:
        # then image is constant image and clip between new_min and new_max
        return np.clip(img, new_min, new_max)
    img_scaled = (img - i_min) / (i_max - i_min) * (new_max - new_min) + new_min
    return img_scaled
