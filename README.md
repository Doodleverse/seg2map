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
  - [ ] Coast Train
  - [ ] FloodNet
  - [ ] Chesapeake Landcover
  - [ ] EnviroAtlas
  - [ ] OpenEarthMap
  - [ ] DeepGlobe
- [ ] Develop codes/docs for selecting model
- [ ] Develop codes/docs for applying model to make label imagery
- [ ] Tool for mosaicing labels
- [ ] [optional] Tool for post-processing/editing labels
- [ ] Tool for downloading labels in geotiff format


