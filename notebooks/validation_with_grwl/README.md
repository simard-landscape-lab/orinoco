# Validation with GRWL

These notebooks represent how we compare segment-level GRWL-derived width and the width from `orinoco`. It is not a totally fair comparison as our segments may not be entirely representative of a localized area particularly around junctions. Nonetheless, we have some indication based on this heuristic comparison that there is good agreement on average.

## Data setup

The GRWL Database can be found [here](https://zenodo.org/record/1297434#.XuUTI2pKgUE). We downloaded a single tile and put them in the following directories:

+ `grwl_data/masks`: water masks 
+ `grwl_data/centerlines`: centerlines with widths

We put each of the tile's data in the relevant folder. In these notebooks, we only look at `NH08`. We could adapt this for any tile, though the subsetting and figures would need adjusting.

### Ocean Mask

We also illustrate how to extract an ocean mask using the [World Water Bodies dataset](https://apps.gis.ucla.edu/geodata/dataset/world_water_bodies/resource/a6b40af0-84cb-40ce-b1c5-b024527a6943). This may be useful for automating this type of deltaic analysis for larger studies across numerous deltas.