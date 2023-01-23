# Seg2Map :mag_right: :milky_way:

*An interactive web map app for applying Doodleverse/Zoo models to geospatial imagery*

![](https://user-images.githubusercontent.com/3596509/194389595-82ade668-daf0-4d24-b1a0-6ecf897f40fe.gif)

Seg2Map will take geospatial imagery from [s2m_engine](https://github.com/Doodleverse/s2m_engine) and apply [Segmentation Zoo](https://github.com/Doodleverse/segmentation_zoo) models to them

Tentative workflow:
* Provide a web map for navigation to a location, and draw a bounding box
* Provide an interface for controls (time period, etc)
* Download geospatial imagery (for now, just NAIP)
* Provide tools to select and apply a Zoo model to create a label image
* Provide tools to interact with those label images (download, mosaic, merge classes, etc)

Roadmap / progress
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
  - [ ] Chesapeake Landcover (NLCD) / NAIP
    - [X] 13 class dataset (no data, water, wetlands, tree canopy, shrubland, low veg, barren, structures, impervious (other), impervious (road), tree canopy over impervious surface, tree canopy over impervious road, other)
    - [ ] set of models
    - [ ] zenodo release  
  - [ ] EnviroAtlas / NAIP 
    - [X] 13 class dataset (no data, water, wetlands, tree canopy, shrubland, low veg, barren, structures, impervious (other), impervious (road), tree canopy over impervious surface, tree canopy over impervious road, other)
    - [ ] set of models
    - [ ] zenodo release  
  - [ ] OpenEarthMap / aerial / high-res. sat
    - [X] 8 class dataset bareland, rangeland, dev., road, tree, water, ag., building)
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


