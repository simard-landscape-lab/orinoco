# Notebooks

These notebooks represent the applications of our RivNet to obtain river networks using map tiles. These notebooks could easily be modified so long as there is:

1. a river/channel mask
2. an interface (represented as a sink mask) - for example an ocean mask (source mask) to indicate where the river runs into the ocean (interface)

The data is included, but we discuss how to obtain the data elsewhere for later use.

## The Notebooks

There are two examples included here using 

1. `stamen_terrain_12` and 
2. `google_16` 

which refer to `<tile_server>_<zoom_level>`. The notebooks navigate through the application and analysis of the river network and the generation of the relevant figures. See below how to obtain the above tif files associated with the tile server.


## Addendum on River Masks

River masks are hard. Shouldn't be a surprise.

It is not our goal here to figure out how to create new water masks, but rather to obtain and utilize existing water masks to generate a network. Although satellite imagery offers high resolution pictures of the current state of a river body, there are numerous challenges creating a useable river mask especially one with high resolution.

We will describe some ways we have generated effective river masks from open data sources that are quick. 


+ There are open tile servers such as Open street map, Stamen, etc. These maps are what are used here. The servers for the pngs and zoom levels for some can be found [here](https://wiki.openstreetmap.org/wiki/Tile_servers). I used the open source tile merger [Stitch](https://github.com/ericfischer/tile-stitch). I am sure there are alternatives such as [this](https://github.com/jimutt/tiles-to-tiff), but stitch worked the best.
	+ **Warning**: Stitch doesn't allow images to be larger than `10,000 x 10,000` - pick the zoom appropriately or be ready to [`gdal_merge.py`](https://gdal.org/programs/gdal_merge.html) the tiles together.
	+ **Warning**: Large downloads are discouraged for open servers.
	+ **Warning**: Can use google maps as well (including satellite) though there are rate limitations, and if you experiment too much, you will be booted for at least 24 hours (in my experience).

	Here are some sample `stitch` commands for obtaining data over the Wax Lake delta areas; the first two are used for the data we use in the notebooks

	google maps: 
```./stitch -f geotiff -w -o google_16.tif -- 29.3745 -91.5573 29.6832 -91.1196 16 https://mt1.google.com/vt/lyrs\=m\&x\=\{x\}\&y\=\{y\}\&z\=\{z\}```

	stamen terrain: 
```./stitch -f geotiff -w -o stamen_terrain_16.tif -- 29.3745 -91.5444 29.6832 -91.0861 14 http://tile.stamen.com/terrain-background/\{z\}/\{x\}/\{y\}.png```

	osm no labels: 
```./stitch -f geotiff -w -o osm_no_labels_16.tif -- 29.3745 -91.5444 29.6832 -91.0861 16 https://maps.wikimedia.org/osm/\{z\}/\{x\}/\{y\}.png```


+ [Peckel](https://global-surface-water.appspot.com/download) - this is as "low-level" as we will get. Basically, this takes the Landsat catalog and determines water in each image. The final products are general statistics about the Landsat water occurrence. An easy way to handle the Peckel data is to download the `occurrence` rasters and select a threshold to find the percentage of pixels in the time series containing water.

	+ **Warning**: Because deltaic areas are lots of swamp land, there are areas in Peckel that appear as perpetually "flooded" particularly in the Mississippi. This makes it hard to determine the rough movement of water around vegetation.

+ [GRWL](https://zenodo.org/record/1297434#.XcywsEVKjUI) - these are from Allen / Pavelsky are Peckel-derived. They too have *Global coverage*!
	+ **Warning**: Tiles are in UTM and each tile has specific UTM zone.
	+ **Warning**: The naming convention for GRWL tiles is not UTM!
	+ **Warning**: Adjacent tiles may not overlap well in that the rivers are disconnected.
	+ **Warning**: Poor coverage over deltaic areas.

	
