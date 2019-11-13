from rasterio.warp import (calculate_default_transform,
                           reproject,
                           Resampling)
import numpy as np
from rasterio.transform import xy
from affine import Affine
from geopy import distance
import pyproj


def _swap(t):
    a, b = t
    return b, a


def project_to_4326(point, source_proj):
    proj_4326 = pyproj.Proj(proj='latlong', datum='WGS84')
    return pyproj.transform(source_proj, proj_4326, *point)


def get_meters_between_points(p1: tuple, p2: tuple):
    """
    p1, p2 in 4326 as (lon, lat)
    """
    q1 = _swap(p1)
    q2 = _swap(p2)
    return distance.distance(q1, q2).meters


def get_dx_dy(profile):
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


def reproject_to_match_profile(src_array: np.ndarray,
                               src_profile: dict,
                               ref_profile: dict,
                               nodata='',
                               resampling='bilinear'):
    height, width = ref_profile['height'], ref_profile['width']
    crs = ref_profile['crs']
    transform = ref_profile['transform']
    count = src_profile['count']

    src_dtype = src_profile['dtype']

    reproject_profile = ref_profile.copy()
    reproject_profile.update({'dtype': src_dtype})

    if nodata is '':
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


def get_cropped_profile(profile: dict, slice_x: slice, slice_y: slice):
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


def get_bounds_dict(profile):
    lx, ly = profile['width'], profile['height']
    transform = profile['transform']
    bounds_dict = {'left': transform.c,
                   'right': transform.c + transform.a * lx,
                   'top': transform.f,
                   'bottom': transform.f + transform.e * ly
                   }
    return bounds_dict


def reproject_to_new_crs(src_array, src_profile, dst_crs):
    reprojected_profile = src_profile.copy()
    bounds_dict = get_bounds_dict(src_profile)

    src_crs = src_profile['crs']
    dst_transform, dst_width, dst_height = calculate_default_transform(src_crs,
                                                                       dst_crs,
                                                                       src_profile['width'],
                                                                       src_profile['height'],
                                                                       **bounds_dict
                                                                       )

    count = src_profile['count']
    if count == 1:
        dst_array = np.zeros((dst_height, dst_width))
    else:
        dst_array = np.zeros((count, dst_height, dst_width))
    # update the relevant parts of the profile
    reprojected_profile.update({
                                'crs': dst_crs,
                                'transform': dst_transform,
                                'width': dst_width,
                                'height': dst_height,
                                })

    reproject(
              # Source parameters
              source=src_array,
              src_crs=src_crs,
              src_transform=src_profile['transform'],
              # Destination paramaters
              destination=dst_array,
              dst_transform=dst_transform,
              dst_crs=dst_crs,
              dst_nodata=src_profile['nodata'],
              # Configuration
              resampling=Resampling.bilinear,
              )
    return dst_array, reprojected_profile
