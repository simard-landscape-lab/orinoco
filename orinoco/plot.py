import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import affine
from typing import Tuple
import matplotlib

####################
# Cartopy Utils
####################


def get_extent(transform: affine.Affine, height: int, width: int, rel_buffer: float = 0.) -> list:
    """
    Get plotting extent as [xmin, xmax, ymin, ymax].

    We assume upper left corner corresponds to transform and width and height
    are in pixels with possible buffer specified by `rel_buffer`.

    Parameters
    ----------
    transform : affine.Affine
        The geographic transform to specify the resolution and upper left corner.
    height : int
        Vertical dimension
    width : int
        Horizontal dimension
    rel_buffer : float
        The relative buffer, i.e. .15 specifies a 15% buffer around all dimensions.

    Returns
    -------
    list:
        [xmin, xmax, ymin, ymax]
    """
    lx, ly = width, height
    x_buffer = transform.a * (rel_buffer) * lx
    y_buffer = transform.e * (rel_buffer) * ly
    extent = [
              # x coordinates (left, right)
              transform.c - x_buffer,
              transform.c + transform.a * ly + x_buffer,
              # y coordinates (top / bottom)
              transform.f - y_buffer,
              transform.f + transform.e * lx + y_buffer,
              ]
    return extent


def make_map(projection: ccrs.PlateCarree = ccrs.PlateCarree(),
             figsize: tuple = (20, 20),
             ticklabelsize: int = 20,
             grid_zorder: int = 3,
             labels: bool = False) -> Tuple[matplotlib.figure.Figure,
                                            matplotlib.axes._subplots.AxesSubplot]:
    """
    Obtaining a fig, ax for cartopy plotting quickly.

    Source: https://stackoverflow.com/questions/49155110/why-do-my-google-tiles-look-poor-in-a-cartopy-map

    Parameters
    ----------
    projection : ccrs.PlateCarree
        The cartopy project
    figsize : tuple
        The tuple specifying figsize in matplotlib
    ticklabelsize : int
        Fontsize of ticklabels with lat, lon
    grid_zorder : int
        The `zorder` layering of grid associated with ticklabels
    labels : bool
        Whether to include the lat, long ticklabels, defaults to False.

    Returns
    -------
    Tuple[matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot]:
        (fig, ax) for cartopy plotting.
    """
    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw=dict(projection=projection))
    gl = ax.gridlines(draw_labels=labels, zorder=grid_zorder)
    if labels:
        gl.xlabels_top = gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': ticklabelsize, 'color': 'black', 'rotation': 'vertical', }
    gl.ylabel_style = {'size': ticklabelsize, 'color': 'black'}
    return fig, ax
