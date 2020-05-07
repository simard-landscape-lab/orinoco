import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import matplotlib.ticker as mticker

####################
# Cartopy Utils
####################


def get_extent(transform, height, width, rel_buffer=0):
    """
    Get plotting extent assuming upper left corner corresponds to transform and width and height
    are in pixels.
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


def make_map(projection=ccrs.PlateCarree(),
             figsize=(20, 20),
             ticklabelsize=20,
             grid_zorder=3,
             x_locs=None,
             labels=False):
    """
    Source: https://stackoverflow.com/questions/49155110/why-do-my-google-tiles-look-poor-in-a-cartopy-map
    """
    fig, ax = plt.subplots(figsize=figsize,
                           subplot_kw=dict(projection=projection))
    gl = ax.gridlines(draw_labels=labels, zorder=grid_zorder)
    if labels:
        gl.xlabels_top = gl.ylabels_right = False
    if x_locs is not None:
        gl.xlocator = mticker.FixedLocator(x_locs)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': ticklabelsize, 'color': 'black', 'rotation': 'vertical', }
    gl.ylabel_style = {'size': ticklabelsize, 'color': 'black'}
    return fig, ax
