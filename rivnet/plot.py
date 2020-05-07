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


def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Source: https://github.com/delestro/rand_cmap
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np

    if type not in ('bright', 'soft'):
        print('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                              boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap
