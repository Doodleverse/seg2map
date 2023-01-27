# Seg2Map :mag_right: :milky_way:

*An interactive web map app for applying Doodleverse/Zoo models to geospatial imagery*

![](https://user-images.githubusercontent.com/3596509/194389595-82ade668-daf0-4d24-b1a0-6ecf897f40fe.gif)

## Overview:
* Seg2Map facilitates application of Deep Learning-based image segmentation models and apply them to high-resolution (~1m or less spatial footprint) geospatial imagery, in order to make high-resolution label maps. 

* The principle aim is to generate time-series of label maps from a time-series of imagery, in order to detect and assess land use/cover change. This project also demonstrates how to apply generic models for land-use/cover on publicly available high-resolution imagery at arbitrary locations.

* Imagery comes from Google Earth Engine via [s2m_engine](https://github.com/Doodleverse/s2m_engine). Initially, we focus on [NAIP](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-aerial-photography-national-agriculture-imagery-program-naip) time-series, available for the conterminious United States since 2003. In the future, [Planetscope](https://developers.planet.com/docs/data/planetscope/) imagery may also be made available (for those with access, such as federal researchers).

* We offer a set of [Segmentation Zoo](https://github.com/Doodleverse/segmentation_zoo) models, especially created and curated for this project based on a publicly available datasets. These datasets have been selected because they are public, large (several hundred to several thousand labeled images), and provide broad class labels for generic land use/cover mapping needs.

## Generic workflow:
* Provide a web map for navigation to a location, and draw a bounding box
* Provide an interface for controls (set time period, etc)
* Download geospatial imagery (for now, just NAIP)
* Provide tools to select and apply a Zoo model to create a label image
* Provide tools to interact with those label images (download, mosaic, merge classes, etc)

## Authors

Package maintainers:
* [@dbuscombe-usgs](https://github.com/dbuscombe-usgs)

Contributions:

* [@2320sharon](https://github.com/2320sharon)
* [@venuswku](https://github.com/venuswku)

We welcome collaboration! Please use our [Discussions](https://github.com/Doodleverse/seg2map/discussions) tab if you're interested in this project. We welcome user-contributed models! They must be trained using [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym), and then served and documented through [Segmentation Zoo](https://github.com/Doodleverse/segmentation_zoo) - get in touch and we'll walk you through the process!

## Roadmap / progress

### V1 
- [X] Develop codes to create a web map for navigation to a location, and draw a bounding box
- [X] Develop codes interface for controls (time period, etc)
- [X] Develop codes for downloading NAIP imagery using GEE
- [ ] Put together a prototype jupyter notebook for web map, bounding box, and image downloads
- [ ] Create Seg2Map models
  - [ ] [Coast Train](https://coasttrain.github.io/CoastTrain/) / aerial / high-res. sat
    - [X] 2 class [dataset](https://coasttrain.github.io/CoastTrain/docs/Version%201:%20March%202022/data) (water, other)
    - [ ] set of models
    - [ ] zenodo release for 768x768 imagery [zenodo page](https://doi.org/10.5281/zenodo.7574784) 
  - [ ] [Coast Train](https://coasttrain.github.io/CoastTrain/) / NAIP
    - [X] 5 class [dataset](https://coasttrain.github.io/CoastTrain/docs/Version%201:%20March%202022/data) (water, whitewater, sediment, bare terrain, other terrain)
    - [X] 8 class [dataset](https://coasttrain.github.io/CoastTrain/docs/Version%201:%20March%202022/data) (water, whitewater, sediment, bare terrain, marsh veg, terrestrial veg, ag., dev.)
    - [ ] set of 5-class models
    - [ ] zenodo release of 5-class models for 768x768 imagery [zenodo page](https://doi.org/10.5281/zenodo.7566992)   
    - [ ] set of 8-class models
    - [ ] zenodo release of 8-class models for 768x768 imagery [zenodo page](https://doi.org/10.5281/zenodo.7570583)   
  - [ ] [FloodNet](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021) / UAV
    - [X] 10 class [dataset](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021) (Background, Building-flooded, Building-non-flooded, Road-flooded, Road-non-flooded, Water, Tree, Vehicle, Pool, Grass)
    - [X] set of models for 768 x 512 imagery
    - [X] zenodo release for 768 x 512 models [zenodo page](https://doi.org/10.5281/zenodo.7566810)
    - [ ] set of models for 1024 x 768 imagery
    - [ ] zenodo release for 1024 x 768 models [zenodo page](https://doi.org/10.5281/zenodo.7566797)
  - [ ] [Chesapeake Landcover](https://lila.science/datasets/chesapeakelandcover) (CCLC) / NAIP
    - [X] 7 class [dataset](https://lila.science/datasets/chesapeakelandcover) (water, tree canopy / forest, low vegetation / field, barren land, impervious (other), impervious (road), no data)
    - [ ] set of models for 512x512 imagery [zenodo page](https://doi.org/10.5281/zenodo.7576904) 
    - [ ] zenodo release  
  - [ ] [EnviroAtlas](https://zenodo.org/record/6268150#.Y9H3vxzMLRZ) / NAIP 
    - [X] 6 class dataset (water, impervious, barren, trees, herbaceous, shrubland)
    - [ ] set of models for 1024x1024 imagery
    - [ ] zenodo release for 1024 x 1024 models [zenodo page](https://doi.org/10.5281/zenodo.7576909)
  - [ ] [OpenEarthMap](https://open-earth-map.org/) / aerial / high-res. sat
    - [X] 9 class [dataset](https://zenodo.org/record/7223446#.Y9IN2BzMLRY) (bareland, rangeland, dev., road, tree, water, ag., building, nodata)
    - [ ] set of models for 512x512 imagery
    - [ ] zenodo release for 512x512 imagery [zenodo page](https://doi.org/10.5281/zenodo.7576894) 
  - [ ] [DeepGlobe](https://arxiv.org/abs/1805.06561) / aerial / high-res. sat
    - [X] 7 class [dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset) (urban, ag., rangeland, forest, water, bare, unknown)
    - [ ] set of models for 512x512 imagery
    - [ ] zenodo release for 512x512 imagery [zenodo page](https://doi.org/10.5281/zenodo.7576898) 
- [ ] Develop codes/docs for selecting model
- [ ] Develop codes/docs for applying model to make label imagery
- [ ] Tool for mosaicing labels
- [ ] Tool for downloading labels in geotiff format

### V2 
- [ ] Tool for post-processing/editing labels
- [ ] Tool for detecting change
- [ ] Make [Planetscope](https://developers.planet.com/docs/data/planetscope/) 3m imagery available via Planet API (federal researchers only)
- [ ] Include additional models/datasets (TBD)

## Datasets

### Chesapeake Landcover
* [webpage](https://lila.science/datasets/chesapeakelandcover)
* Zenodo model release (512x512): 

### Coast Train
* [paper](https://www.nature.com/articles/s41597-023-01929-2)
* [website](https://coasttrain.github.io/CoastTrain/)
* [data](https://cmgds.marine.usgs.gov/data-releases/datarelease/10.5066-P91NP87I/)
* [preprint](https://eartharxiv.org/repository/view/3560/)
* Zenodo model release, 2-class (768x768): 
* Zenodo model release, 5-class (768x768): 
* Zenodo model release, 8-class (768x768): 

### DeepGlobe
* [paper](https://arxiv.org/abs/1805.06561)
* [challenge](http://deepglobe.org/challenge.html)
* [data](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)
* Zenodo model release (512x512): 

### EnviroAtlas
* [EnviroAtlas dataset](https://zenodo.org/record/6268150#.Y9H3vxzMLRZ)
* [EnviroAtlas paper](https://www.mdpi.com/2072-4292/12/12/1909)
* [paper using EnviroAtlasdata](https://arxiv.org/pdf/2202.14000.pdf)
* This dataset was organized to accompany the 2022 paper, [Resolving label uncertainty with implicit generative models](https://openreview.net/forum?id=AEa_UepnMDX). More details can be found [here](https://github.com/estherrolf/qr_for_landcover)
* Zenodo model release (1024x1024): 

### FloodNet
* [FloodNet dataset](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021)
* [challenge](http://www.classic.grss-ieee.org/earthvision2021/challenge.html)
* [paper](https://arxiv.org/abs/2012.02951)
* Zenodo model release (768x512): Buscombe, Daniel. (2023). Doodleverse/Segmentation Zoo/Seg2Map Res-UNet models for FloodNet/10-class segmentation of RGB 768x512 UAV images (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7566810
* Zenodo model release (1024x768): 

### OpenEarthMap
* [website](https://open-earth-map.org/)
* [data](https://zenodo.org/record/7223446#.Y7zQLxXMK3A)
* [paper](https://arxiv.org/abs/2210.10732)
* Zenodo model release (512x512): 


## Classes

Superclasses:

A. Water
B. Sediment
C. Bare
D. Vegetated
E. Impervious

| | Coast Train 1 |  Coast Train 2 |  Coast Train 3| FloodNet | Chesapeake| EnviroAtlas| OpenEarthMap| DeepGlobe|
|---|---|---|---|---|---|---|---|---|
|A. Water | X| X|X |X |X |X |X |X |
|a. whitewater | |X |X | | | | | | 
|a. pool | | | |X | | | | |
|---|---|---|---|---|---|---|---|---|
|B. Sediment | | X|X | | | | | | 
|C. Bare/barren| |X |X | |X |X | X| X| 
|---|---|---|---|---|---|---|---|---|
|d. marsh | | |X | | | | | | 
|d. terrestrial veg| | |X | | | | | |
|d. agriculture| | | X| | | |X | X|
|d. grass | | | |X | | | | |
|d. herbaceous / low vegetation / field | | | | | X|X | | |
|d. tree/forest | | | |X |X |X | X|X |
|d. shrubland | | | | | |X | | |
|d. rangeland | | | | | |X | X| X|
|---|---|---|---|---|---|---|---|---|
|E. Impervious/urban/developed | | |X | | |X | X| X|
|e. impervious (other) | | | | |X | | | |
|e. impervious (road) | | | | |X | | X| |
|e. Building-flooded | | | | X| | | | |
|e. Building-non-flooded | | | |X | | |X | |
|e. Road-flooded | | | |X | | | | |
|e. Road-non-flooded | | | |X | | | | |
|e. Vehicle | | | |X | | | | |
|---|---|---|---|---|---|---|---|---|
|X. Other | X| X| | | | | | |


## References

### Notes
* [NLCD classes](https://www.mrlc.gov/data/legends/national-land-cover-database-class-legend-and-description)
* [NAIP imagery](https://doi.org/10.5066/F7QN651G)