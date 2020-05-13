# Orinoco

>Orinoco is a delta in Venezuela and literally means “a place to paddle” in Warao, but refers to a place being navigable.

Orinoco is a tool to leverage python GIS tools to generate river networks from a water mask and a source/sink area (e.g. the ocean).

Products:

+ A networkx directed Graph that can be used for additional analysis.
+ River Centroids and Centerlines (corresponding to the network's nodes and edges, respectively)

We generate the river network using scikit-fmm, skimage, networkx, geopandas, shapely, etc. See example in [notebooks](notebooks) to see how these products are generated.

# Dependencies

+ rasterio
+ skimage
+ networkx
+ geopandas
+ numpy
+ scipy
+ tqdm
+ matplotlib
+ shapely
+ jupyter (for examples)

## Installation

Follow the instructions [here](https://github.com/cmarshak/sari-tutorial) and then additionally install:

+ networkx

I would just use pip for the above, but do what you like (it's a pure python library). To install rivnet. Navigate to the directory containing this and in your terminal:

`pip install .`

