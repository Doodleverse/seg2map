# Seg2Map :mag_right: :milky_way:

*An interactive web map app for applying Doodleverse/Zoo models to geospatial imagery*

![](https://user-images.githubusercontent.com/3596509/194389595-82ade668-daf0-4d24-b1a0-6ecf897f40fe.gif)

Seg2Map will take geospatial imagery from [s2m_engine](https://github.com/Doodleverse/s2m_engine) and apply [Segmentation Zoo](https://github.com/Doodleverse/segmentation_zoo) models to them

## Workflow:
* Provide a web map for navigation to a location, and draw a bounding box
* Provide an interface for controls (time period, etc)
* Download geospatial imagery (for now, just NAIP)
* Provide tools to select and apply a Zoo model to create a label image
* Provide tools to interact with those label images (download, mosaic, merge classes, etc)

## Roadmap / progress
- [X] Develop codes to create a web map for navigation to a location, and draw a bounding box
- [X] Develop codes interface for controls (time period, etc)
- [X] Develop codes for downloading NAIP imagery using GEE
- [ ] Put together a prototype jupyter notebook for web map, bounding box, and image downloads
- [ ] Create Seg2Map models
  - [ ] Coast Train / aerial / high-res. sat
    - [X] 2 class dataset (water, other)
    - [ ] set of models
    - [ ] zenodo release  
  - [ ] Coast Train / NAIP
    - [X] 5 class dataset (water, whitewater, sediment, bare terrain, other terrain)
    - [ ] set of models
    - [ ] zenodo release  
  - [ ] Coast Train / NAIP
    - [X] 8 class dataset (water, whitewater, sediment, bare terrain, marsh veg, terrestrial veg, ag., dev.)
    - [ ] set of models
    - [ ] zenodo release  
  - [ ] FloodNet / UAV
    - [X] 10 class dataset (Background, Building-flooded, Building-non-flooded, Road-flooded, Road-non-flooded, Water, Tree, Vehicle, Pool, Grass)
    - [ ] set of models
    - [ ] zenodo release
  - [ ] Chesapeake Landcover (CCLC) / NAIP
    - [X] 7 class dataset (water, tree canopy / forest, low vegetation / field, barren land, impervious (other), impervious (road), no data)
    - [ ] set of models
    - [ ] zenodo release  
  - [ ] [EnviroAtlas](https://zenodo.org/record/6268150#.Y9H3vxzMLRZ) / NAIP 
    - [X] 6 class dataset (water, impervious, barren, trees, herbaceous, shrubland)
    - [ ] set of models
    - [ ] zenodo release  
  - [ ] OpenEarthMap / aerial / high-res. sat
    - [X] 9 class dataset (bareland, rangeland, dev., road, tree, water, ag., building, nodata)
    - [ ] set of models
    - [ ] zenodo release  
  - [ ] DeepGlobe / aerial / high-res. sat
    - [X] 7 class dataset (urban, ag., rangeland, forest, water, bare, unknown)
    - [ ] set of models
    - [ ] zenodo release  
- [ ] Develop codes/docs for selecting model
- [ ] Develop codes/docs for applying model to make label imagery
- [ ] Tool for mosaicing labels
- [ ] [optional] Tool for post-processing/editing labels
- [ ] Tool for downloading labels in geotiff format


## References

### Chesapeake Landcover
* https://lila.science/datasets/chesapeakelandcover
*

### Coast Train
* paper https://www.nature.com/articles/s41597-023-01929-2
* website https://coasttrain.github.io/CoastTrain/
* data https://cmgds.marine.usgs.gov/data-releases/datarelease/10.5066-P91NP87I/
* preprint https://eartharxiv.org/repository/view/3560/ 

### DeepGlobe
* DeepGlobe https://arxiv.org/abs/1805.06561 
* DeepGlobe http://deepglobe.org/challenge.html
* DEEPGLOBE - 2018 Satellite Challange http://deepglobe.org/index.html

### EnviroAtlas
* [EnviroAtlas dataset](https://zenodo.org/record/6268150#.Y9H3vxzMLRZ)
* [EnviroAtlas paper](https://www.mdpi.com/2072-4292/12/12/1909)
* [paper using EnviroAtlasdata](https://arxiv.org/pdf/2202.14000.pdf)
* This dataset was organized to accompany the 2022 paper, [Resolving label uncertainty with implicit generative models](https://openreview.net/forum?id=AEa_UepnMDX). More details can be found [here](https://github.com/estherrolf/qr_for_landcover)

### FloodNet
* FloodNet https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021 
* http://www.classic.grss-ieee.org/earthvision2021/challenge.html 

### OpenEarthMap
* Open Earth Map https://open-earth-map.org/
* Open Earth Map https://zenodo.org/record/7223446#.Y7zQLxXMK3A 
* Open Earth Map https://arxiv.org/abs/2210.10732 

### Notes
* NLCD https://www.mrlc.gov/data/legends/national-land-cover-database-class-legend-and-description