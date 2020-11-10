# Orinoco

>Orinoco is a delta in Venezuela and literally means “a place to paddle” in Warao, but refers to a place being navigable.

Orinoco is a tool to leverage python GIS tools to generate river networks from a water mask (in UTM) and a source/sink area (e.g. the ocean). We generate the river network using scikit-fmm, skimage, networkx, geopandas, shapely, etc. See example in [notebooks](notebooks) to see how these products are generated.

Our journal article discussing the methodology and validation can be found [here](https://www.mdpi.com/2220-9964/9/11/658).

>> Marshak, C.; Simard, M.; Denbina, M.; Nilsson, J.; Van der Stocken, T. Orinoco: Retrieving a River Delta Network with the Fast Marching Method and Python. ISPRS Int. J. Geo-Inf. 2020, 9, 658.

## Products and Demonstrations

### Products

+ A networkx directed Graph that can be used for additional analysis.
+ River Centroids and Centerlines (corresponding to the network's nodes and edges, respectively)

![example](example.png)


### Notebooks

In our jupyter notebooks, we demontrate:

+ how to use Stamen Terrain tiles and Google map tiles, create a simple water mask over the Wax Lake and Atachfalaya River Deltas
+ how to obtain the channel network and related products with the software tools here
+ how to compute the normalized entropy rate over the subnetwork related to the Wax Lake Outlet
+ how to recreate our validation using the Global River Widths from Landsat (GRWL) Database
+ Validating widths using GRWL water masks and USGS gauges.	


# Installation

The installations are based on python 3.7+ using anaconda and conda within a virtual environment. Generally, this is accomplished as follows:

1. Create a virtual environment using the `requirements.txt`
2. Install orinoco downloading this repo and `pip install .`

More explicitly, using the anaconda distribution for Mac or Windows:

1. `conda create --name orinoco_env python=3.7`
2. `conda activate orinoco_env`
3. `conda install -c conda-forge --yes --file requirements.txt`
4. `pip install .`
5. Ensure your python can be found by jupyter via `python -m ipykernel install --user`

You can make sure your installation was done correctly running `python -c "import orinoco"` and/or running the notebooks.

# Example Notebooks

Open up a jupyter notebook (using `jupyter-notebook`) and navigate to `notebooks/`. 

1. All the data for [`notebooks/examples`](notebooks/examples) is available *in this repository* including the merged png tiles over the Wax Lake and Atchafalaya Rivers. You should be able to run the notebooks (in order) without modification.

2. The [`notebooks/validation_with_grwl`](notebooks/validation_with_grwl) reproduces the GRWL comparison from our [article](https://www.mdpi.com/2220-9964/9/11/658). To reproduce the GRWL comparison, you will have to download some data and make sure the notebooks reference their location appropriately on your local machine. We used tile NH08 from [GRWL Database](https://zenodo.org/record/1297434#.XuK6hWpKgUE). We also use the World Water Body Dataset from [here](https://apps.gis.ucla.edu/geodata/dataset/world_water_bodies/resource/a6b40af0-84cb-40ce-b1c5-b024527a6943) as an initialization mask.

3. There are notebooks in [`notebooks/validation_with_rivgraph`](notebooks/validation_with_rivgraph) which compares our products to those products in the RivGraph journal [article](https://esurf.copernicus.org/articles/8/87/2020/). The data is included in this repository though you will again need the [World Water Body Dataset](https://apps.gis.ucla.edu/geodata/dataset/world_water_bodies/resource/a6b40af0-84cb-40ce-b1c5-b024527a6943). The instructions are described in the notebooks.
4. There are notebooks for [comparisons with usgs](notebooks/validation_with_usgs) again using the GRWL masks. You will need to download GRWL Tiles NH15 as above from the [GRWL Database](https://zenodo.org/record/1297434#.XuK6hWpKgUE). The USGS station data can be obtained using the notebooks pulling data from USGS APIs.

# General Philosophy

Our adaptation of the fast-marching method provides a new tool for understanding deltaic connectivity, but there is still a lot that can be done and improved upon. It's hard to imagine this tool alone capturing all the connectivity and channel directivity perfectly. As such, we anticipate some of the best parts of our approach will be combined with some of the approaches we cite below (or synthesized into improved methodologies) in order to provide a more comprehensive view of deltaic connectivity.

This principle is also reflected in our rather simple and flat design of `orinoco`, namely, as a collection of functions modifying networkx Graphs and writing to GeoDataFrames. It is not our intention for this work to be a monolith of classes, properties, and instance functions. Rather, we hope what is presented can easily be modified and improved upon for large scale studies.

In other words, remote sensing is hard!

# Related Work

Below are related projects and rough checklist of the products that can be extracted.

+ RivWidth [[repo](http://uncglobalhydrology.org/rivwidth/)][[paper](https://ieeexplore.ieee.org/document/4382932)] - IDL

	- [x] Widths
	- [x] Centerlines
	- [ ] Graph Structure

+ RivGraph [[repo](https://github.com/jonschwenk/RivGraph)] [[paper](https://esurf.copernicus.org/articles/8/87/2020/)] [[poster](https://www.researchgate.net/publication/329845073_Automatic_Extraction_of_Channel_Network_Topology_RivGraph)] - python
	
	- [x] Widths
	- [x] Centerlines
	- [x] Graph Structure


+ Rivamap [[repo](https://github.com/isikdogan/rivamap)][[paper](http://www.isikdogan.com/files/isikdogan2017_rivamap.pdf)] - python

	- [x] Widths
	- [x] Centerlines
	- [ ] Graph Structure

+ CMGO [[repo](https://github.com/AntoniusGolly/cmgo/tree/e9a4dbc286aff17c4d344988f0f9d8350128ce27)][[paper](https://esurf.copernicus.org/articles/5/557/2017/esurf-5-557-2017.html)] - R

    - [x] Widths
	- [x] Centerlines
	- [ ] Graph Structure 

+ ChanGeom [[repo](https://www.burchfisher.com/data.html)][[paper](https://www.burchfisher.com/uploads/3/8/8/3/38838315/fisher_etal_geomorph_2013.pdf)] - Matlab

	- [x] Widths
	- [x] Centerlines
	- [ ] Graph Structure

+ Centerline [[repo](https://github.com/fitodic/centerline)] - python

	- [ ] Widths
	- [x] Centerlines
	- [ ] Graph Structure



# License

See [LICENSE.txt](LICENSE.txt).

>Copyright 2020, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.

>This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.