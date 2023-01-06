
# pass entire geodataframe
# get the ids of the selected ROIs that will be downloaded
# for each roi split them into a series of equally sized tiles
# each tile's area <= 10km^2
import json
import math
import os, json, shutil
from glob import glob

import requests
from typing import List, Tuple
import platform
import tqdm
import tqdm.auto
import zipfile
from area import area
import numpy as np
import geopandas as gpd
import asyncio
import aiohttp
import tqdm.asyncio
import nest_asyncio
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
    print(f"tile_path: {tile_path}")
    if not os.path.exists(tile_path):
        os.makedirs(tile_path)
        
    collection = ee.ImageCollection(gee_collection)
    polys = ee.Geometry.Polygon(tile)
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
 
def create_dir(dir_path:str,raise_error=True)->str:
    dir_path=os.path.abspath(dir_path)
    if os.path.exists(dir_path):
        if raise_error:
            raise FileExistsError(dir_path)
    else:
        os.makedirs(dir_path)
    return dir_path
 
def copy_multiband_tifs(roi_path:str,multiband_path:str ):
    for folder in glob(
        roi_path
        + os.sep
        + "chunk*",
        recursive=True,
    ):
        files = glob(folder + os.sep + "*multiband.tif")
        [
            shutil.copyfile(file, multiband_path + os.sep + file.split(os.sep)[-1])
            for file in files
        ] 
        
   
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
    print(f"Area: {area_km2}")
    if area_km2 <= 1:
        return 0 
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

def create_time_series_dirs(sitename:str,years:List[int],num_features:int):
    for featurenumber in range(num_features):
        for year in years:
            dir_path = os.path.abspath(sitename + os.sep + "feature" + str(featurenumber))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            sub_dir=os.path.join(dir_path,str(year))
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

async def fetch_all(session, tile_coords,site_path:str,gee_collection:str,dates:Tuple[str]):
    # creates task for each tile to be downloaded and waits for tasks to complete
    tasks = []
    for counter,tile in enumerate(tile_coords):
        task = asyncio.create_task(async_download_tile(session,tile,site_path,counter,gee_collection,dates))
        tasks.append(task)
    await tqdm.asyncio.tqdm.gather(*tasks,position=0,desc=f"All Downloads")

async def async_download_tiles(tiles:MultiPolygon,site_path:str,gee_collection:str,dates:Tuple[str]):
    tile_coords= [list(part.exterior.coords) for part in tiles.geoms]
    print("Number of individual tiles: {}".format(len(tile_coords)))
    async with aiohttp.ClientSession() as session:
        await fetch_all(session, tile_coords,site_path,gee_collection,dates)

def run_async_download(tiles:MultiPolygon,site_path:str,gee_collection:str,dates:Tuple[str]):
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # apply a nested loop to jupyter's event loop for async downloading
    # nest_asyncio.apply()
    # get nested running loop and wait for async downloads to complete
    loop = asyncio.get_running_loop()
    result = loop.run_until_complete(async_download_tiles(tiles,site_path,gee_collection,dates))

async def async_ee_export_image(
    session,
    counter,
    ee_object,
    filename,
    scale=None,
    crs=None,
    crs_transform=None,
    region=None,
    dimensions=None,
    file_per_band=False,
    format="ZIPPED_GEO_TIFF",
    unmask_value=None,
    timeout=300,
    proxies=None,
):
    """Exports an ee.Image as a GeoTIFF.
    Args:
        ee_object (object): The ee.Image to download.
        filename (str): Output filename for the exported image.
        scale (float, optional): A default scale to use for any bands that do not specify one; ignored if crs and crs_transform is specified. Defaults to None.
        crs (str, optional): A default CRS string to use for any bands that do not explicitly specify one. Defaults to None.
        crs_transform (list, optional): a default affine transform to use for any bands that do not specify one, of the same format as the crs_transform of bands. Defaults to None.
        region (object, optional): A polygon specifying a region to download; ignored if crs and crs_transform is specified. Defaults to None.
        dimensions (list, optional): An optional array of two integers defining the width and height to which the band is cropped. Defaults to None.
        file_per_band (bool, optional): Whether to produce a different GeoTIFF per band. Defaults to False.
        format (str, optional):  One of: "ZIPPED_GEO_TIFF" (GeoTIFF file(s) wrapped in a zip file, default), "GEO_TIFF" (GeoTIFF file), "NPY" (NumPy binary format). If "GEO_TIFF" or "NPY",
            filePerBand and all band-level transformations will be ignored. Loading a NumPy output results in a structured array.
        unmask_value (float, optional): The value to use for pixels that are masked in the input image.
            If the exported image contains zero values, you should set the unmask value to a  non-zero value so that the zero values are not treated as missing data. Defaults to None.
        timeout (int, optional): The timeout in seconds for the request. Defaults to 300.
        proxies (dict, optional): A dictionary of proxy servers to use. Defaults to None.
    """

    if not isinstance(ee_object, ee.Image):
        print("The ee_object must be an ee.Image.")
        return

    if unmask_value is not None:
        ee_object = ee_object.selfMask().unmask(unmask_value)
        if isinstance(region, ee.Geometry):
            ee_object = ee_object.clip(region)
        elif isinstance(region, ee.FeatureCollection):
            ee_object = ee_object.clipToCollection(region)

    filename = os.path.abspath(filename)
    basename = os.path.basename(filename)
    name = os.path.splitext(basename)[0]
    filetype = os.path.splitext(basename)[1][1:].lower()
    filename_zip = filename.replace(".tif", ".zip")

    if filetype != "tif":
        print("The filename must end with .tif")
        return

    try:
        # print("Generating URL ...")
        params = {"name": name, "filePerBand": file_per_band}

        params["scale"] = scale
        if region is None:
            region = ee_object.geometry()
        if dimensions is not None:
            params["dimensions"] = dimensions
        if region is not None:
            params["region"] = region
        if crs is not None:
            params["crs"] = crs
        if crs_transform is not None:
            params["crs_transform"] = crs_transform
        params["format"] = format

        try:
            url = ee_object.getDownloadURL(params)
        except Exception as e:
            print(e)
            print("An error occurred while downloading.")
            raise e
            return
        # print(f"Downloading data from {url}\nPlease wait ...")
        chunk_size: int = 2048
        async with session.get(url,timeout=timeout, raise_for_status=True) as r:
            if r.status != 200:
                print("An error occurred while downloading.")
                print(r.status)
                return
            content_length = r.headers.get("Content-Length")
            if content_length is not None:
                content_length = int(content_length)
                with open(filename_zip, "wb") as fd:
                    with tqdm.auto.tqdm(
                        total=content_length,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading tile {counter}",
                        initial=0,
                        ascii=False,
                        position=counter,
                    ) as pbar:
                        async for chunk in r.content.iter_chunked(chunk_size):
                            fd.write(chunk)
                            pbar.update(len(chunk))
            else:
                with open(filename_zip, "wb") as fd:
                    async for chunk in r.content.iter_chunked(chunk_size):
                        fd.write(chunk)           
    except Exception as e:
        print(e)
        print("An error occurred while downloading.")
        return

    try:
        with zipfile.ZipFile(filename_zip) as z:
            z.extractall(os.path.dirname(filename))
        # os.remove(filename_zip)

        if file_per_band:
            print(f"Data downloaded to {os.path.dirname(filename)}")
        else:
            print(f"Data downloaded to {filename}")
    except Exception as e:
        print(e)
     
async def async_download_tile(
    session,
    tile:List[Tuple],
                  site_path:str,
                  counter:int,
                  gee_collection:str,
                  dates:Tuple[str]):

    OUT_RES_M = 0.5  # output raster spatial footprint in metres
    # full path to directory where downloaded images will be saved
    tile_path=os.path.join(site_path,"chunk"+ str(counter))
    if not os.path.exists(tile_path):
        os.makedirs(tile_path)
    # instantiate a gee collection and download specific region within date range
    collection = ee.ImageCollection(gee_collection)
    polys = ee.Geometry.Polygon(tile)
    # get portion of collection within tile
    collection = collection.filterBounds(polys)
    # only get images within the start and end date range
    collection = collection.filterDate(dates[0], dates[1]).sort(
                "system:time_start", True
            )
    # create list of image names within collection
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
        await async_ee_export_image(
            session,
            counter,
            image,
            tif_path,
            scale=OUT_RES_M,
            region=polys,
            file_per_band=True,
            crs="EPSG:4326",)
        # export single image with all bands
        await async_ee_export_image(
            session,
            counter,
            image,
            multiband_path,
            scale=OUT_RES_M,
            region=polys,
            file_per_band=False,                       
            crs="EPSG:4326",
        )

def merge_tifs(tifs_path:str,roi_path:str)->str:
    # run gdal to merge into big tiff
    # ensure path to ROI exists
    if not os.path.exists(roi_path):
        raise FileNotFoundError(roi_path)
    # ensure path containing all the tifs exists
    if not os.path.exists(tifs_path):
        raise FileNotFoundError(tifs_path)
    try:
        ## create vrt
        vrtoptions = gdal.BuildVRTOptions(
            resampleAlg="average", srcNodata=0, VRTNodata=0
        )
        files = glob(tifs_path + os.sep + "*multiband.tif")
        # full path to save vrt (virtual world format) file
        vrt_path = os.path.join(roi_path,"merged_multispectral.vrt")
        print(f"outfile for vrt: {vrt_path}")
        # creates a virtual world file using all the tifs 
        # if vrt file already exists it is overwritten
        virtual_dataset = gdal.BuildVRT(vrt_path, files, options=vrtoptions)
        # flushing the cache causes the vrt file to be created 
        virtual_dataset.FlushCache()
        # reset the dataset object 
        virtual_dataset = None
        # create geotiff (.tiff) from merged vrt file
        virtual_dataset = gdal.Translate(
            vrt_path.replace(".vrt", ".tif"),
            creationOptions=["COMPRESS=LZW", "TILED=YES"],
            srcDS=vrt_path,
        )
        virtual_dataset.FlushCache()
        virtual_dataset = None
        # convert .vrt to .jpg file
        # uses gdal to create a world file
        virtual_dataset = gdal.Translate(
            vrt_path.replace(".vrt", ".jpg"),
            creationOptions=["WORLDFILE=YES", "QUALITY=100"],
            srcDS=vrt_path.replace(".vrt", ".tif"),
        )
        virtual_dataset.FlushCache()
        virtual_dataset = None
        return vrt_path
    except Exception as e:
        print(e)
        print("Data not available")
        raise e