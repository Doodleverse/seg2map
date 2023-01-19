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
from roi_func import *

def create_dir(dir_path: str, raise_error=True) -> str:
    dir_path = os.path.abspath(dir_path)
    if os.path.exists(dir_path):
        if raise_error:
            raise FileExistsError(dir_path)
    else:
        os.makedirs(dir_path)
    return dir_path

# download tiles functions must create all filepaths first
# then fetch all can be called

async def async_download_tile(
    session, polygon, tile_id: str,
    filepath: str,
    counter:int,
    filePerBand: bool
):

    OUT_RES_M = 0.5  # output raster spatial footprint in metres
    image_ee = ee.Image(tile_id)
    print(image_ee)
    print(f"polygon: {polygon}")
    # crop and download
    download_id = ee.data.getDownloadId(
        {
            "image": image_ee,
            "region": polygon,
            "scale": OUT_RES_M,
            "crs": "EPSG:4326",
            "filePerBand": filePerBand,
        }
    )
    try:
        # create download url using id
        # response = requests.get(ee.data.makeDownloadUrl(download_id))
        fp_zip = os.path.join(filepath, tile_id.replace('/','_') + ".zip")
        # fp_zip = os.path.join(filepath, "temp.zip")
        # with open(fp_zip, "wb") as fd:
        #     fd.write(response.content)
        # # unzip the individual bands
        
        # if file_per_band:
        #     print(f"Data downloaded to {os.path.dirname(filename)}")
        # else:
        #     print(f"Data downloaded to {filename}")

        timeout = 300
        chunk_size: int = 2048
        # create download url using id
        url = ee.data.makeDownloadUrl(download_id)
        # zip directory images will be downloaded to
        async with session.get(url, timeout=timeout, raise_for_status=True) as r:
            if r.status != 200:
                print("An error occurred while downloading.")
                print(r.status)
                return
            content_length = r.headers.get("Content-Length")
            if content_length is not None:
                content_length = int(content_length)
                with open(fp_zip, "wb") as fd:
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
                with open(fp_zip, "wb") as fd:
                    async for chunk in r.content.iter_chunked(chunk_size):
                        fd.write(chunk)
        # unzip the downloaded file
        with zipfile.ZipFile(fp_zip) as local_zipfile:
            for fn in local_zipfile.namelist():
                local_zipfile.extract(fn, filepath)
        os.remove(fp_zip)


    except Exception as e:
        print(e)
        raise e

def mk_filepaths(imgs_dict: List[dict]):
    filepaths = [img_dict["filepath"] for img_dict in imgs_dict]
    for filepath in filepaths:
        multiband_filepath = os.path.join(filepath, "multiband")
        create_dir(filepath, raise_error=False)
        create_dir(multiband_filepath, raise_error=False)

async def fetch_all(session, imgs_dict: List[dict]):
    # creates task for each tile to be downloaded and waits for tasks to complete
    tasks = []
    # GEE allows for 10 concurrent requests at once
    MAX_REQUESTS = 10
    requests_left = MAX_REQUESTS
    for counter, tile_dict in enumerate(imgs_dict):
        # make two requests for each image
        polygon = tile_dict["polygon"]
        filepath = os.path.abspath(tile_dict["filepath"])
        multiband_filepath = os.path.join(filepath, "multiband")

        for tile_id in tile_dict["ids"]:
            # have under 10 requests been made?
            if requests_left > 0:
                # schedule to download multiband
                task = asyncio.create_task(
                    async_download_tile(
                        session,
                        polygon,
                        tile_id,
                        multiband_filepath,
                        counter,
                        filePerBand=False
                    )
                )
                tasks.append(task)
                # schedule to download all bands
                task = asyncio.create_task(
                    async_download_tile(
                        session,
                        polygon,
                        tile_id,
                        filepath,
                        counter,
                        filePerBand=True
                    )
                )
                tasks.append(task)
                requests_left -= 2
            # if all available tasks have been scheduled and tasks are avaiable wait for the downloads to complete
            if requests_left <= 0 and len(tasks) > 0:
                await tqdm.asyncio.tqdm.gather(
                    *tasks, position=0, desc=f"All Downloads"
                )
                requests_left = MAX_REQUESTS
                tasks = []

    # Download remaining ids after all tiles have been processed
    if len(tasks) != 0:
        await tqdm.asyncio.tqdm.gather(*tasks, position=0, desc=f"All Downloads")
        requests_left = MAX_REQUESTS
        tasks = []


async def async_download_tiles(tiles_info:List[dict]):
    async with aiohttp.ClientSession() as session:
        await fetch_all(session, tiles_info)

def run_async_download(tiles_info:List[dict]):
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # apply a nested loop to jupyter's event loop for async downloading
    # nest_asyncio.apply()
    # get nested running loop and wait for async downloads to complete
    loop = asyncio.get_running_loop()
    result = loop.run_until_complete(async_download_tiles(tiles_info))


# new method
gee_collection = "USDA/NAIP/DOQQ"
ee.Initialize()

filename = r"C:\1_USGS\5_Doodleverse\1_Seg2Map_fork\seg2map\src\seg2map\NAIP\hidden_beach.geojson"
with open(filename, "r") as f:
    gpd_data = gpd.read_file(f)

# get number of splitters need to split ROI into rectangles of 1km^2 area (or less)
num_splitters = get_num_splitters(gpd_data)
print(num_splitters)
# split ROI into rectangles of 1km^2 area (or less)
split_polygon = splitPolygon(gpd_data, num_splitters)
tile_coords = [list(part.exterior.coords) for part in split_polygon.geoms]
# print(f"tile_coords: {tile_coords}")

# SINGLE LARGE ROI
# -------------------------------------------------------------
# only works if gpd_data is a geodataframe with a single entry
# split_polygon = Polygon(gpd_data["geometry"].iloc[0])
# print(f"split_polygon {split_polygon}")
# polygon_geom = split_polygon.envelope
# print(f"polygon_geom {polygon_geom}")
# coords_polygon = np.array(polygon_geom.exterior.coords)
# print(f"coords_polygon {coords_polygon}")
# print(list(split_polygon.exterior.coords))
# tile_coords = [list(split_polygon.exterior.coords)]
# print(f"tile_coords {tile_coords}")
# -------------------------------------------------------------


dates = ("2010-01-01", "2010-12-31")
dir_path = os.path.abspath(os.getcwd() + os.sep + "ROIs")
roi_path = os.path.abspath(os.getcwd() + os.sep + "ROIs"+ os.sep + "ROI3")
if os.path.exists(roi_path):
    shutil.rmtree(roi_path)


def get_tiles_info(
    tile_coords: list, dates: Tuple[str], roi_path: str, gee_collection: str
) -> List[dict]:
    sum_imgs = 0
    all_img_ids = []
    img_dicts = []
    for counter, tile in enumerate(tile_coords):
        collection = ee.ImageCollection(gee_collection)
        polys = ee.Geometry.Polygon(tile)
        # get portion of collection within tile and date range
        collection = collection.filterBounds(polys)
        collection = collection.filterDate(dates[0], dates[1]).sort(
            "system:time_start", True
        )
        # make a list of all the image names in the collection
        im_list = collection.getInfo().get("features")
        ids = [obj["id"] for obj in im_list]
        all_img_ids.extend(ids)
        # for each tile put the geometry, tile directory path, and filenames to download in a dict
        image_dict = {
            "polygon": tile,
            "ids": ids,
            "filepath": roi_path + os.sep + str(counter),
        }
        img_dicts.append(image_dict)
        # print(f"\nids: {ids}")
        print(f"\n Images available for tile {counter} : {len(im_list)}")
        sum_imgs += len(im_list)

    print(f"\nTotal Images available across all tiles {sum_imgs}")
    return img_dicts


# get list of tiles info needed to downloaded all tiles
tiles_info = get_tiles_info(tile_coords, dates, roi_path, gee_collection)
# make directories for all tiles within ROI
mk_filepaths(tiles_info)
# download all the tiles within the ROI
# run_async_download(tiles_info)
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# download the tiles that compose the ROI in parallel
asyncio.run(async_download_tiles(tiles_info))

