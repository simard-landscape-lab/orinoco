# Notebooks

River masks are hard. It is not our goal to generate new masks here, but rather use existing masks to generate a network.

We will describe some ways we have generated effective river masks from open data sources.

+ [GRWL](https://zenodo.org/record/1297434#.XcywsEVKjUI) - these are from Allen / Pavelsky and are derived from the well-known [Peckel](https://global-surface-water.appspot.com/download) data. Peckel in-turn is derived from the Landsat catalog. *Global coverage*!
	+ **Warning**: Tiles are in UTM and each tile has specific UTM zone.
	+ **Warning**: The naming convention for GRWL tiles is not UTM!
	+ **Warning**: Adjacent tiles may not overlap well in that the rivers are disconnected .
+ There are open tile servers such as Open street map, Stamen, etc. The servers for the pngs and zoom levels for some can be found [here](https://wiki.openstreetmap.org/wiki/Tile_servers). I used the open source tile merger [Stitch](https://github.com/ericfischer/tile-stitch). I am sure there are alternatives such as [this](https://github.com/jimutt/tiles-to-tiff), but stitch worked the best.
	+ **Warning**: Stitch doesn't allow images to be larger than 10,000 x 10,000 - pick the zoom appropriately or be ready to stitch together all images that stitch gives you.
	+ **Warning**: Large downloads are discouraged for open servers.
	+ **Warning**: Can use google maps as well (including satellite) though there are strict rate limitations.

Here are some sample `stitch` commands for obtaining data over the Wax Lake delta.

google maps: 
```./stitch -f geotiff -w -o google_16.tif -- 29.3745 -91.5444 29.6832 -91.0861 16 https://mt1.google.com/vt/lyrs\=m\&x\=\{x\}\&y\=\{y\}\&z\=\{z\}```

stamen terrain: 
```./stitch -f geotiff -w -o stamen_terrain_16.tif -- 29.3745 -91.5444 29.6832 -91.0861 14 http://tile.stamen.com/terrain-background/\{z\}/\{x\}/\{y\}.png```

osm no labels: 
```./stitch -f geotiff -w -o osm_no_labels_16.tif -- 29.3745 -91.5444 29.6832 -91.0861 16 https://maps.wikimedia.org/osm/\{z\}/\{x\}/\{y\}.png```