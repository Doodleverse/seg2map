# Seg2Map :mag_right: :milky_way:

*An interactive web map app for applying Doodleverse/Zoo models to geospatial imagery*

![](https://user-images.githubusercontent.com/3596509/194389595-82ade668-daf0-4d24-b1a0-6ecf897f40fe.gif)

## Overview:
* Seg2Map facilitates application of Deep Learning-based image segmentation models and apply them to high-resolution (~1m or less spatial footprint) geospatial imagery, in order to make high-resolution label maps. Please see our [wiki](https://github.com/Doodleverse/seg2map/wiki) for more information.

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
* [@dbuscombe-usgs](https://github.com/dbuscombe-usgs)
* [@2320sharon](https://github.com/2320sharon)

Contributions:
* [@venuswku](https://github.com/venuswku)

We welcome collaboration! Please use our [Discussions](https://github.com/Doodleverse/seg2map/discussions) tab if you're interested in this project. We welcome user-contributed models! They must be trained using [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym), and then served and documented through [Segmentation Zoo](https://github.com/Doodleverse/segmentation_zoo) - get in touch and we'll walk you through the process!

## Roadmap / progress

### V1 
- [X] Develop codes to create a web map for navigation to a location, and draw a bounding box
- [X] Develop codes interface for controls (time period, etc)
- [X] Develop codes for downloading NAIP imagery using GEE
- [X] Put together a prototype jupyter notebook for web map, bounding box, and image downloads
- [ ] Create Seg2Map models
  - [X] [Coast Train](https://coasttrain.github.io/CoastTrain/) / aerial / high-res. sat
    - [X] 2 class [dataset](https://coasttrain.github.io/CoastTrain/docs/Version%201:%20March%202022/data) (water, other)
    - [X] zenodo release for 768x768 imagery [zenodo page](https://doi.org/10.5281/zenodo.7574784) 
  - [X] [Coast Train](https://coasttrain.github.io/CoastTrain/) / NAIP
    - [X] 5 class [dataset](https://coasttrain.github.io/CoastTrain/docs/Version%201:%20March%202022/data) (water, whitewater, sediment, bare terrain, other terrain)
    - [X] 8 class [dataset](https://coasttrain.github.io/CoastTrain/docs/Version%201:%20March%202022/data) (water, whitewater, sediment, bare terrain, marsh veg, terrestrial veg, ag., dev.)
    - [X] zenodo release of 5-class ResUNet models for 768x768 imagery [zenodo page](https://doi.org/10.5281/zenodo.7566992)   
    - [X] zenodo release of 8-class ResUNet models for 768x768 imagery [zenodo page](https://doi.org/10.5281/zenodo.7570583)   
    - [X] zenodo release of 5-class Segformer models for 768x768 imagery [zenodo page](https://doi.org/10.5281/zenodo.7641708)   
    - [X] zenodo release of 8-class Segformer models for 768x768 imagery [zenodo page](https://doi.org/10.5281/zenodo.7641724)      
  - [X] [Chesapeake Landcover](https://lila.science/datasets/chesapeakelandcover) (CCLC) / NAIP
    - [X] 7 class [dataset](https://lila.science/datasets/chesapeakelandcover) (water, tree canopy / forest, low vegetation / field, barren land, impervious (other), impervious (road), no data)
    - [X] zenodo release of 7-class ResUNet models for 512x512 imagery [page](https://doi.org/10.5281/zenodo.7576904)
    - [X] zenodo release of 7-class SegFormer models for 512x512 imagery [page](https://doi.org/10.5281/zenodo.7677506)    
  - [X] [EnviroAtlas](https://zenodo.org/record/6268150#.Y9H3vxzMLRZ) / NAIP 
    - [X] 6 class dataset (water, impervious, barren, trees, herbaceous, shrubland)
    - [X] zenodo release of 6-class ResUNet models for 1024 x 1024 models [zenodo page](https://doi.org/10.5281/zenodo.7576909)
  - [X] [OpenEarthMap](https://open-earth-map.org/) / aerial / high-res. sat
    - [X] 9 class [dataset](https://zenodo.org/record/7223446#.Y9IN2BzMLRY) (bareland, rangeland, dev., road, tree, water, ag., building, nodata)
    - [X] zenodo release of 9-class ResUNet models for 512x512 models [zenodo page](https://doi.org/10.5281/zenodo.7576894) 
  - [X] [DeepGlobe](https://arxiv.org/abs/1805.06561) / aerial / high-res. sat
    - [X] 7 class [dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset) (urban, ag., rangeland, forest, water, bare, unknown)
    - [X] zenodo release for of 7-class ResUNet models 512x512 imagery [zenodo page](https://doi.org/10.5281/zenodo.7576898) 
  - [ ] [Barrier Islands](https://www.sciencebase.gov/catalog/item/5d5ece47e4b01d82ce961e36) / orthomosaic / coastlines
    - [ ] Substrate data 6 class (dev, sand, mixed, coarse, unknown, water)
    - [ ] zenodo release of substrate models for 768x768 imagery  
    - [ ] Vegetation type data 7 class (shrub/forest, shrub, none/herb., none, herb., herb./shrub, dev)
    - [ ] zenodo release of Vegetation type  models for 768x768 imagery   
    - [ ] Vegetation density data 7 class (dense, dev., moderate, moderate/dense, none, none/sparse, sparse)
    - [ ] zenodo release of Vegetation density  models for 768x768 imagery  
    - [X] Geomorphic setting data 7 class (beach, backshore, dune, washover, barrier interior, marsh, ridge/swale)
    - [ ] zenodo release of Geomorphic setting  models for 768x768 imagery  
    - [X] Supervised classification data 9 class (water, sand, herbaceous veg./low shrub, sparse/moderate, herbaceous veg/low shrub, moderate/dense, high shrub/forest, marsh/sediment, marsh/veg, marsh, high shrub/forest, development)
    - [ ] zenodo release of Supervised classification models for 768x768 imagery      
  - [X] [AAAI](https://github.com/FrontierDevelopmentLab/multi3net) / aerial / high-res. sat
    - [X] 2 class dataset (other, building)
    - [X] zenodo release for 1024x1024 imagery [zenodo page](https://doi.org/10.5281/zenodo.7607895)
    - [X] 2 class dataset (other, flooded building)
    - [X] zenodo release for 1024x1024 imagery [zenodo page](https://doi.org/10.5281/zenodo.7613106)
  - [X] xBD-hurricanes / aerial / high-res. sat, a subset of the [XView2](https://xview2.org/) dataset
    - [X] 4 class building dataset (other, no damage, minor damage, major damage)
    - [X] zenodo release for 768x768 imagery [zenodo page](https://doi.org/10.5281/zenodo.7613175)
    - [X] 2 class building dataset (other, building)
    - [X] zenodo release for 768x768 imagery [zenodo page](https://doi.org/10.5281/zenodo.7613212)
  - [ ] Superclass models
    - [X] 8 merged datasets for 8 separate superclass models (water, sediment, veg, herb. veg., woody veg., impervious, building, agriculture)
    - [ ] zenodo release for 768x768 imagery / water
    - [ ] zenodo release for 768x768 imagery / sediment
    - [ ] zenodo release for 768x768 imagery / veg
    - [ ] zenodo release for 768x768 imagery / herb. veg.
    - [ ] zenodo release for 768x768 imagery / woody veg.
    - [ ] zenodo release for 768x768 imagery / impervious
    - [ ] zenodo release for 768x768 imagery / building
    - [ ] zenodo release for 768x768 imagery / agriculture    
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

### General Landcover

#### DeepGlobe
* [paper](https://arxiv.org/abs/1805.06561)
* [challenge](http://deepglobe.org/challenge.html)
* [data](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)
* Zenodo model release (512x512): Buscombe, Daniel. (2023). Doodleverse/Segmentation Zoo/Seg2Map Res-UNet models for DeepGlobe/7-class segmentation of RGB 512x512 high-res. images (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7576898

#### EnviroAtlas
* [EnviroAtlas dataset](https://zenodo.org/record/6268150#.Y9H3vxzMLRZ)
* [EnviroAtlas paper](https://www.mdpi.com/2072-4292/12/12/1909)
* [paper using EnviroAtlasdata](https://arxiv.org/pdf/2202.14000.pdf)
* This dataset was organized to accompany the 2022 paper, [Resolving label uncertainty with implicit generative models](https://openreview.net/forum?id=AEa_UepnMDX). More details can be found [here](https://github.com/estherrolf/qr_for_landcover)
* Zenodo model release (512x512): Buscombe, Daniel. (2023). Doodleverse/Segmentation Zoo/Seg2Map Res-UNet models for EnviroAtlas/6-class segmentation of RGB 512x512 high-res. images (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7576909

#### OpenEarthMap
* [website](https://open-earth-map.org/)
* [data](https://zenodo.org/record/7223446#.Y7zQLxXMK3A)
* [paper](https://arxiv.org/abs/2210.10732)
* Zenodo model release (512x512): Buscombe, Daniel. (2023). Doodleverse/Segmentation Zoo/Seg2Map Res-UNet models for OpenEarthMap/9-class segmentation of RGB 512x512 high-res. images (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7576894


### Coastal Landcover

#### Chesapeake Landcover
* [webpage](https://lila.science/datasets/chesapeakelandcover)
* Zenodo model release (512x512): Buscombe, Daniel. (2023). Doodleverse/Segmentation Zoo/Seg2Map Res-UNet models for Chesapeake/7-class segmentation of RGB 512x512 high-res. images (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7576904
* Zenodo SegFormer model release (512x512): Buscombe, Daniel. (2023). Doodleverse/Segmentation Zoo/Seg2Map Segformer models for Chesapeake/7-class segmentation of RGB 512x512 high-res. images (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7677506

#### Coast Train
* [paper](https://www.nature.com/articles/s41597-023-01929-2)
* [website](https://coasttrain.github.io/CoastTrain/)
* [data](https://cmgds.marine.usgs.gov/data-releases/datarelease/10.5066-P91NP87I/)
* [preprint](https://eartharxiv.org/repository/view/3560/)
* Zenodo model release, 2-class (768x768): Buscombe, Daniel. (2023). Doodleverse/Segmentation Zoo/Seg2Map Res-UNet models for CoastTrain water/other segmentation of RGB 768x768 orthomosaic images (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7574784
* Zenodo model release, 5-class (768x768): Buscombe, Daniel. (2023). Doodleverse/Segmentation Zoo/Seg2Map Res-UNet models for CoastTrain/5-class segmentation of RGB 768x768 NAIP images (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7566992
* Zenodo model release, 8-class (768x768): Buscombe, Daniel. (2023). Doodleverse/Segmentation Zoo/Seg2Map Res-UNet models for CoastTrain/8-class segmentation of RGB 768x768 NAIP images (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7570583
* Zenodo SegFormer model release, 5-class (768x768): Buscombe, Daniel. (2023). Doodleverse/Segmentation Zoo/Seg2Map SegFormer models for CoastTrain/5-class segmentation of RGB 768x768 NAIP images (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7641708
* Zenodo SegFormer model release, 8-class (768x768): Buscombe, Daniel. (2023). Doodleverse/Segmentation Zoo/Seg2Map SegFormer models for CoastTrain/8-class segmentation of RGB 768x768 NAIP images (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7641724

#### AAAI / Buildings / Flooded Buildings
* [data](https://github.com/FrontierDevelopmentLab/multi3net)
* [data](https://github.com/orion29/Satellite-Image-Segmentation-for-Flood-Damage-Analysis)
* [paper](https://arxiv.org/pdf/1812.01756.pdf)
* Zenodo model release (1024x1024) building / no building: Buscombe, Daniel. (2023). Doodleverse/Segmentation Zoo/Seg2Map Res-UNet models for segmentation of buildings of RGB 1024x1024 high-res. images (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7607895
* Zenodo model release (1024x1024) flooded building / no flooded building: Buscombe, Daniel. (2023). Doodleverse/Segmentation Zoo/Seg2Map Res-UNet models for segmentation of AAAI/flooded buildings in RGB 1024x1024 high-res. images (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7622733

#### XBD-hurricanes
* [Xview2 challenge](https://xview2.org/)
* [XBD-hurricanes code](https://github.com/MARDAScience/XBD-hurricanes)
* Zenodo SegFormer model release (768x768) building damage: Buscombe, Daniel. (2023). Doodleverse/Segmentation Zoo/Seg2Map SegFormer models for segmentation of xBD/damaged buildings in RGB 768x768 high-res. images (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7613175
* Zenodo SegFormer model release (768x768) building presence/absence: Buscombe, Daniel. (2023). Doodleverse/Segmentation Zoo/Seg2Map SegFormer models for segmentation of xBD/buildings in RGB 768x768 high-res. images (v1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7613212

#### Barrier Islands
* [webpage](https://www.sciencebase.gov/catalog/item/5d5ece47e4b01d82ce961e36)
* [paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0209986)
* Zenodo Substrate model release (768x768): 
* Zenodo Vegetation type model release (768x768):
* Zenodo Vegetation density model release (768x768):
* Zenodo Geomorphic model release (768x768):
* Zenodo Supervised classification model release (768x768):


## Superclasses

A. Water:
  * Coast Train
  * Chesapeake
  * EnviroAtlas
  * OpenEarthMap
  * DeepGlobe
  * Barrier Substrate
  * NOAA
  * [v2: Barrier Substrate]
  * [v2: Elwha]

B. Sediment:
  * Coast Train
  * NOAA
  * [v2: Barrier Substrate (sand, mixed, coarse)]
  * [v2: Elwha]

C. Bare:
  * Chesapeake (barren land)
  * EnviroAtlas (barren)
  * OpenEarthMap (bareland)

D. Vegetated:
  * Coast Train (marsh veg, terrestrial veg, ag)
  * FloodNet (tree, grass)
  * Chesapeake (tree canopy / forest, low vegetation / field)
  * EnviroAtlas (trees, herbaceous, shrubland)
  * OpenEarthMap (rangeland, tree, ag)
  * DeepGlobe (ag., rangeland, forest)
  * NOAA (veg)
  * [v2: Elwha]  
  * [v2: Barrier Substrate]

E. Impervious:
  * FloodNet (Building-flooded, Building-non-flooded, Road-flooded, Road-non-flooded, vehicle)
  * Chesapeake (impervious (other), impervious (road))
  * EnviroAtlas (impervious)
  * OpenEarthMap (dev, road, building)
  * DeepGlobe (urban)
  * NOAA (dev)
  * [v2: Elwha]

F. Building:
  * OpenEarthMap (building)
  * AAAI (building)

G. Agriculture:
  * OpenEarthMap (ag)
  * DeepGlobe (ag)

H. Woody Veg:
  * FloodNet (tree)
  * Chesapeake (tree canopy / forest)
  * EnviroAtlas (trees)
  * OpenEarthMap (tree)
  * DeepGlobe (forest)
  * [v2: Elwha]  
  * [v2: Barrier Substrate]

## References

### Notes
* [NLCD classes](https://www.mrlc.gov/data/legends/national-land-cover-database-class-legend-and-description)
* [NAIP imagery](https://doi.org/10.5066/F7QN651G)


## Classes: 

| | Coast Train 1 |  Coast Train 2 |  Coast Train 3| FloodNet | Chesapeake| EnviroAtlas| OpenEarthMap| DeepGlobe| AAAI | NOAA | Barrier Substrate  |
|---|---|---|---|---|---|---|---|---|---|---|---|
|A. Water | X| X|X |X |X |X |X |X | | X|X|
|a. whitewater | |X |X | | | | | | | | |
|a. pool | | | |X | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
|B. Sediment | | X|X | | | | | | | X| |
|b. sand | | | | | | | | | | |X|
|b. mixed | | | | | | | | | | | X|
|b. coarse | | | | | | | | | | | X|
|---|---|---|---|---|---|---|---|---|---|---|---|
|C. Bare/barren| |X |X | |X |X | X| X| | | |
|---|---|---|---|---|---|---|---|---|---|---|
|d. marsh | | |X | | | | | | | | |
|d. terrestrial veg| | |X | | | | | | | X| |
|d. agriculture| | | X| | | |X | X| | | |
|d. grass | | | |X | | | | | | | |
|d. herbaceous / low vegetation / field | | | | | X|X | | | | | |
|d. tree/forest | | | |X |X |X | X|X | | | |
|d. shrubland | | | | | |X | | | | | |
|d. rangeland | | | | | |X | X| X| | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
|E. Impervious/urban/developed | | |X | | |X | X| X| | X| X|
|e. impervious (other) | | | | |X | | | | | | |
|e. impervious (road) | | | | |X | | X| | | | |
|e. Building-flooded | | | | X| | | | | | | |
|e. Building-non-flooded | | | |X | | |X | | X| | |
|e. Road-flooded | | | |X | | | | | | | |
|e. Road-non-flooded | | | |X | | | | | | | |
|e. Vehicle | | | |X | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
|X. Other | X| X| | | | | | | X| | |

