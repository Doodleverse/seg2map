
# pass entire geodataframe
# get the ids of the selected ROIs that will be downloaded
# for each roi split them into a series of equally sized tiles
# each tile's area <= 10km^2
import json
import math
import os, json, shutil
from glob import glob
from typing import List, Tuple

from area import area
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import split
import geemap, ee
from osgeo import gdal

def download_tile(tile:List[Tuple],
                  site_path:str,
                  counter:int,
                  gee_collection:str,
                  dates:Tuple[str]):
    # full path where downloaded image will be saved
    OUT_RES_M = 0.5  # output raster spatial footprint in metres
    
    tile_path=os.path.join(site_path,"chunk"+ str(counter))
    if os.path.exists(tile_path):
        os.makedirs(tile_path)
        
    collection = ee.ImageCollection(gee_collection)
    polys = ee.Geometry.Polygon(tile)
    # get center of polygon as longitude and latitude
    centroid = polys.centroid()
    lng, lat = centroid.getInfo()["coordinates"]
    # get portion of collection within tile
    collection = collection.filterBounds(polys)
    # only get images within the start and end date range
    collection = collection.filterDate(dates[0], dates[1]).sort(
                "system:time_start", True
            )
    count = collection.size().getInfo()
    img_lst = collection.toList(1000)
    img_names = []
    for i in range(0, count):
        image = ee.Image(img_lst.get(i))
        name = image.get("system:index").getInfo()
        img_names.append(name)
        print(name)
    # download each of the images from the collection    
    for name in img_names:
        image = ee.Image(gee_collection+"/" + name)
        # full path where tif file will be saved
        tile_path=os.path.join(site_path,"chunk"+ str(counter))
        # make the folder to hold the chunks
        if not os.path.exists(tile_path):
            os.makedirs(tile_path)
        tif_path = os.path.join(tile_path,"chunk"+ str(counter)+ "_"+ name+ ".tif")
        multiband_path= os.path.join(tile_path,"chunk"+ str(counter)+ "_"+ name+ "_multiband.tif")
        # Export each band as one image
        geemap.ee_export_image(
            image,
            tif_path,
            scale=OUT_RES_M,
            region=polys,
            file_per_band=True,
            crs="EPSG:4326",)
        # export single image with all bands
        geemap.ee_export_image(
            image,
            multiband_path,
            scale=OUT_RES_M,
            region=polys,
            file_per_band=False,
            crs="EPSG:4326",
        )

def download_tiles(tiles:MultiPolygon,site_path:str,gee_collection:str,dates:Tuple[str]):
    tile_coords= [list(part.exterior.coords) for part in tiles.geoms]
    print("Number of individual tiles: {}".format(len(tile_coords)))
    for counter,tile in enumerate(tile_coords):
        download_tile(tile,site_path,counter,gee_collection,dates)
    
def create_site_dirs(sitename:str,years:List[str],num_features:int):
    """
    Create directories for a site with multiple features, organized by year.
    
    Parameters:
    sitename (str): name of site.
    years (List[str]): years for which directories should be created.
    num_features (int): number of features for which directories should be created.
    
    Returns:
    None
    
    Example:
    >>> create_site_dirs("Site1", ["2020", "2021"], 3)
    """
    for featurenumber in range(num_features):
        for year in years:
            dir_path = os.path.abspath(sitename + os.sep + "feature" + str(featurenumber))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            sub_dir=os.path.join(dir_path,str(year))
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
  
def get_num_splitters(gdf:gpd.GeoDataFrame)->int:
    """
    Calculates the minimum number of splitters required to divide a geographic region represented by a GeoDataFrame into smaller, equal-sized tiles whose
    area <= 1 km^2.
    
    max area per tile is 1 km^2
    
    Parameters:
        gdf (gpd.GeoDataFrame): A GeoDataFrame representing the geographic region to be split. Must contain only a single entry.

    Returns:
        int: An integer representing the minimum number of splitters required to divide the region represented by `gdf` into smaller, equal-sized tiles whose
        area <= 1 km^2.

    """  
    # convert to geojson dictionary
    roi_json = json.loads(gdf.to_json())
    # only one feature is present select 1st feature's geometry
    roi_geometry = roi_json['features'][0]['geometry']
    # get area of entire shape as squared kilometers 
    area_km2=area(roi_geometry)/ 1e6
    # get minimum number of horizontal and vertical splitters to split area equally
    # max area per tile is 1 km^2
    num_splitters = math.ceil(math.sqrt(area_km2))
    return num_splitters

def splitPolygon(polygon:gpd.GeoDataFrame, num_splitters:int)->MultiPolygon:
    """
    Split a polygon into a given number of smaller polygons by adding horizontal and vertical lines.
    
    Parameters:
    polygon (gpd.GeoDataFrame): A GeoDataFrame object containing a single polygon.
    num_splitters (int): The number of horizontal and vertical lines to add.
    
    Returns:
    MultiPolygon: A MultiPolygon object containing the smaller polygons.
    
    Example:
    >>> import geopandas as gpd
    >>> from shapely.geometry import Polygon, MultiPolygon
    >>> poly = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
    >>> df = gpd.GeoDataFrame(geometry=[poly])
    >>> result = splitPolygon(df, 2)
    >>> result # polygon split into 4 equally sized tiles
    """
    minx, miny, maxx, maxy = polygon.bounds.iloc[0]
    dx = (maxx - minx) / num_splitters  # width of a small part
    dy = (maxy - miny) / num_splitters  # height of a small part
    horizontal_splitters = [LineString([(minx, miny + i*dy), (maxx, miny + i*dy)]) for i in range(num_splitters)]
    vertical_splitters = [LineString([(minx + i*dx, miny), (minx + i*dx, maxy)]) for i in range(num_splitters)]
    splitters = horizontal_splitters + vertical_splitters
    result = polygon["geometry"].iloc[0]
    for splitter in splitters:
        result = MultiPolygon(split(result, splitter))
    return result

