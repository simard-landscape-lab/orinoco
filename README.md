# RivNet

A tool to leverage python GIS tools to generate river networks from a water mask and a sink area (e.g. the ocean).

Products:

+ A networkx directed Graph that can be used for additional analysis.
+ River Centroids and Centerlines (corresponding to the networks nodes and edges)

We generate the river network using scikit-fmm, skimage, networkx. See example in [notebooks](notebooks) to see how these products are generated.

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