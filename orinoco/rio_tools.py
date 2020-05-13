from rasterio.warp import calculate_default_transform, reproject, Resampling, aligned_target
from rasterio.transform import xy
from affine import Affine
from rasterio import features
from rasterio.features import shapes
import numpy as np
import fiona
from geopy import distance
import pyproj
from typing import Union, Tuple


def _swap(t: tuple):
    a, b = t
    return b, a


def project_to_4326(point, source_proj):
    proj_4326 = pyproj.Proj(proj='latlong', datum='WGS84')
    return pyproj.transform(source_proj, proj_4326, *point)


def get_meters_between_4326_points(p1: tuple, p2: tuple):
    """
    p1, p2 in 4326 as (lon, lat)
    """
    q1 = _swap(p1)
    q2 = _swap(p2)
    return distance.distance(q1, q2).meters


def get_4326_dx_dy(profile):
    t = profile['transform']
    crs = str(profile['crs']).lower()
    source_proj = pyproj.Proj(init=crs)

    def project_partial(p): return project_to_4326(p, source_proj)

    # dx
    p0 = (t * (0, 0))
    p1 = (t * (1, 0))
    p0_4326, p1_4326 = project_partial(p0), project_partial(p1)
    dx = distance.distance(_swap(p0_4326), _swap(p1_4326)).meters

    # dy
    p0 = (t * (0, 0))
    p1 = (t * (0, 1))
    p0_4326, p1_4326 = project_partial(p0), project_partial(p1)
    dy = distance.distance(_swap(p0_4326), _swap(p1_4326)).meters

    return dx, dy


def polygonize_array_to_shapefile(arr, profile, shape_file_dir, label_name='label', mask=None, connectivity=8):

    if mask is None:
        mask = np.ones(arr.shape).astype(bool)
    else:
        # mask is data mask
        mask = ~mask.astype(bool)

    dtype = str(arr.dtype)
    if 'int' in dtype or 'bool' in dtype:
        arr = arr.astype('int32')
        dtype = 'int32'
        dtype_for_shape_file = 'int'

    if 'float' in dtype:
        arr = arr.astype('float32')
        dtype = 'float32'
        dtype_for_shape_file = 'float'

    transform = profile['transform']
    crs = profile['crs']
    features = list(shapes(arr, mask=mask, transform=transform, connectivity=connectivity))
    results = list({'properties': {label_name: (value)}, 'geometry': geometry} for i, (geometry, value) in enumerate(features))
    with fiona.open(shape_file_dir, 'w',
                    driver='ESRI Shapefile',
                    crs=crs,
                    schema={'properties': [(label_name, dtype_for_shape_file)],
                            'geometry': 'Polygon'}) as dst:
        dst.writerecords(results)


def rasterize_shapes_to_array(shapes: list,
                              attributes: list,
                              profile: dict,
                              all_touched=False) -> np.ndarray:

    """
    Rasterizers a list of shapes and burns them into array with given attributes.
    """
    out_arr = np.zeros((profile['height'], profile['width']))

    # this is where we create a generator of geom, value pairs to use in rasterizing
    shapes = [(geom, value) for geom, value in zip(shapes, attributes)]
    burned = features.rasterize(shapes=shapes,
                                out=out_arr,
                                transform=profile['transform'],
                                all_touched=all_touched)

    return burned


def reproject_arr_to_match_profile(src_array: np.ndarray,
                                   src_profile: dict,
                                   ref_profile: dict,
                                   nodata=None,
                                   resampling='bilinear') -> Tuple[np.ndarray, dict]:
    """
    Note: src_array needs to be in gdal (i.e. BIP) format
    """
    height, width = ref_profile['height'], ref_profile['width']
    crs = ref_profile['crs']
    transform = ref_profile['transform']
    count = src_profile['count']

    src_dtype = src_profile['dtype']

    reproject_profile = ref_profile.copy()
    reproject_profile.update({'dtype': src_dtype})

    if nodata is None:
        nodata = src_profile['nodata']
    reproject_profile.update({'nodata': nodata,
                              'count': count})

    dst_array = np.zeros((count, height, width))

    resampling = Resampling[resampling]

    reproject(src_array,
              dst_array,
              src_transform=src_profile['transform'],
              src_crs=src_profile['crs'],
              dst_transform=transform,
              dst_crs=crs,
              dst_nodata=nodata,
              resampling=resampling)
    return dst_array.astype(src_dtype), reproject_profile


def get_cropped_profile(profile: dict, slice_x: slice, slice_y: slice) -> dict:
    """
    slice_x and slice_y are numpy slices
    """
    x_start = slice_x.start or 0
    y_start = slice_y.start or 0
    x_stop = slice_x.stop or profile['width']
    y_stop = slice_y.stop or profile['height']

    width = x_stop - x_start
    height = y_stop - y_start

    profile_cropped = profile.copy()

    trans = profile['transform']
    x_cropped, y_cropped = xy(trans, y_start, x_start, offset='ul')
    trans_list = list(trans.to_gdal())
    trans_list[0] = x_cropped
    trans_list[3] = y_cropped
    tranform_cropped = Affine.from_gdal(*trans_list)
    profile_cropped['transform'] = tranform_cropped

    profile_cropped['height'] = height
    profile_cropped['width'] = width

    return profile_cropped


def get_bounds_dict(profile: dict) -> dict:
    lx, ly = profile['width'], profile['height']
    transform = profile['transform']
    bounds_dict = {'left': transform.c,
                   'right': transform.c + transform.a * lx,
                   'top': transform.f,
                   'bottom': transform.f + transform.e * ly
                   }
    return bounds_dict


def reproject_profile_to_new_crs(src_profile: dict,
                                 dst_crs: str,
                                 target_resolution: Union[float, int] = None) -> dict:
    reprojected_profile = src_profile.copy()
    bounds_dict = get_bounds_dict(src_profile)

    src_crs = src_profile['crs']
    dst_transform, dst_width, dst_height = calculate_default_transform(src_crs,
                                                                       dst_crs,
                                                                       src_profile['width'],
                                                                       src_profile['height'],
                                                                       **bounds_dict
                                                                       )

    if target_resolution is not None:
        dst_transform, dst_width, dst_height = aligned_target(dst_transform, dst_width, dst_height, target_resolution)
    reprojected_profile.update({
                                'crs': dst_crs,
                                'transform': dst_transform,
                                'width': dst_width,
                                'height': dst_height,
                                })
    return reprojected_profile


def reproject_arr_to_new_crs(src_array: np.ndarray,
                             src_profile: dict,
                             dst_crs: str,
                             resampling: str = 'bilinear',
                             target_resolution: float = None) -> Tuple[np.ndarray, dict]:
    reprojected_profile = reproject_profile_to_new_crs(src_profile, dst_crs, target_resolution=target_resolution)
    resampling = Resampling[resampling]
    dst_array = np.zeros((reprojected_profile['count'], reprojected_profile['height'], reprojected_profile['width']))

    reproject(
              # Source parameters
              source=src_array,
              src_crs=src_profile['crs'],
              src_transform=src_profile['transform'],
              # Destination paramaters
              destination=dst_array,
              dst_transform=reprojected_profile['transform'],
              dst_crs=reprojected_profile['crs'],
              dst_nodata=src_profile['nodata'],
              # Configuration
              resampling=resampling,
              )
    return dst_array, reprojected_profile


def convert_4326_to_utm(lon: float, lat: float) -> str:
    """From: https://gis.stackexchange.com/a/269552
    """
    utm_band = str(int((np.floor((lon + 180) / 6) % 60) + 1))
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    return f'epsg:{epsg_code}'
