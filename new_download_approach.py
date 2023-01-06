# pass entire geodataframe
# get the ids of the selected ROIs that will be downloaded
# for each roi split them into a series of equally sized tiles
# each tile's area <= 10km^2
import json
import math
import os, json, shutil
from glob import glob
import threading
import concurrent.futures

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
from roi_func import splitPolygon,get_num_splitters
from time import perf_counter
start = perf_counter()

with open("multithreading_timer.txt", "w") as f:
    f.write(f"Start Time {start}")


# Initialize a semaphore object with a limit of 10
# GEE allows for 10 concurrent requests at once
limit = asyncio.Semaphore(11)


def remove_zip(path):
    # Get a list of all the zipped files in the directory
    zipped_files = [
        os.path.join(path, f) for f in os.listdir(path) if f.endswith(".zip")
    ]
    # Remove each zip file
    for zipped_file in zipped_files:
        os.remove(zipped_file)


def unzip(path):
    # Get a list of all the zipped files in the directory
    zipped_files = [
        os.path.join(path, f) for f in os.listdir(path) if f.endswith(".zip")
    ]
    # Unzip each file
    for zipped_file in zipped_files:
        with zipfile.ZipFile(zipped_file, "r") as zip_ref:
            zip_ref.extractall(path)


def unzip_files(paths):
    # Create a thread pool with a fixed number of threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit a unzip task for each directory
        futures = [executor.submit(unzip, path) for path in paths]

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)


def remove_zip_files(paths):
    # Create a thread pool with a fixed number of threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit a remove_zip task for each directory
        futures = [executor.submit(remove_zip, path) for path in paths]

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)


def get_subdirs(parent_dir):
    # Get a list of all the subdirectories in the parent directory
    subdirs = [
        os.path.join(parent_dir, d)
        for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]
    return subdirs


def unzip_data(parent_dir: str):
    subdirs = get_subdirs(parent_dir)
    unzip_files(subdirs)
    # remove_zip_files(subdirs)


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
    session,
    polygon,
    tile_id: str,
    filepath: str,
    counter: int,
    img_count: int,
    filename: str,
    filePerBand: bool,
):
    # No more than 10 concurrent workers will be able to make
    # get request at the same time.
    async with limit:
        print(f"limit: {limit}")
        OUT_RES_M = 0.5  # output raster spatial footprint in metres
        image_ee = ee.Image(tile_id)
        # crop and download
        # print(f"filename: {filename}")
        if filename is None:
            download_id = ee.data.getDownloadId(
                {
                    "image": image_ee,
                    "region": polygon,
                    "scale": OUT_RES_M,
                    "crs": "EPSG:4326",
                    "filePerBand": filePerBand,
                }
            )
        else:
            download_id = ee.data.getDownloadId(
                {
                    "image": image_ee,
                    "region": polygon,
                    "scale": OUT_RES_M,
                    "crs": "EPSG:4326",
                    "filePerBand": filePerBand,
                    "name": filename,
                }
            )
        try:
            # create download url using id
            # response = requests.get(ee.data.makeDownloadUrl(download_id))
            if filename is not None:
                fp_zip = os.path.join(filepath, filename+tile_id.replace("/", "_") + ".zip")
            elif filename is None:
                fp_zip = os.path.join(filepath, tile_id.replace("/", "_") + ".zip")
            timeout = 300
            
            chunk_size: int = 2048
            # create download url using id
            url = ee.data.makeDownloadUrl(download_id)

            # When workers hit the limit, they'll wait for a second
            # before making more requests.
            if limit.locked():
                print("Concurrency limit reached, waiting ...")
                await asyncio.sleep(1)

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
                            desc=f"Downloading tile {counter} #{img_count}",
                            initial=0,
                            ascii=False,
                            position=img_count,
                        ) as pbar:
                            async for chunk in r.content.iter_chunked(chunk_size):
                                fd.write(chunk)
                                pbar.update(len(chunk))
                else:
                    with open(fp_zip, "wb") as fd:
                        async for chunk in r.content.iter_chunked(chunk_size):
                            fd.write(chunk)
            # # unzip the downloaded file
            # with zipfile.ZipFile(fp_zip) as local_zipfile:
            #     for fn in local_zipfile.namelist():
            #         local_zipfile.extract(fn, filepath)
            # os.remove(fp_zip)
        except Exception as e:
            print(e)
            raise e


def copy_multiband_tifs(roi_path: str, multiband_path: str):
    for folder in glob(
        roi_path + os.sep + "tile*",
        recursive=True,
    ):
        files = glob(folder + os.sep + "*multiband.tif")
        [
            shutil.copyfile(file, multiband_path + os.sep + file.split(os.sep)[-1])
            for file in files
        ]


def merge_tifs(multiband_path: str, roi_path: str) -> str:
    # run gdal to merge into big tiff
    # ensure path to ROI exists
    if not os.path.exists(roi_path):
        raise FileNotFoundError(roi_path)
    # ensure path containing all the tifs exists
    if not os.path.exists(multiband_path):
        raise FileNotFoundError(multiband_path)
    try:
        ## create vrt
        vrtoptions = gdal.BuildVRTOptions(
            resampleAlg="average", srcNodata=0, VRTNodata=0
        )
        files = glob(multiband_path + os.sep + "*.tif")
        # full path to save vrt (virtual world format) file
        vrt_path = os.path.join(roi_path, "merged_multispectral.vrt")
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


def mk_filepaths(tiles_info: List[dict]):
    filepaths = [tile_info["filepath"] for tile_info in tiles_info]
    for filepath in filepaths:
        create_dir(filepath, raise_error=False)


async def fetch_all(session, imgs_dict: List[dict], sum_imgs: int):
    # creates task for each tile to be downloaded and waits for tasks to complete
    tasks = []
    img_count = 0
    for counter, tile_dict in enumerate(imgs_dict):
        # make two requests for each image
        polygon = tile_dict["polygon"]
        filepath = os.path.abspath(tile_dict["filepath"])
        # ROI directory containing all tile subdirectories
        parent_dir = os.path.dirname(filepath)
        multiband_filepath = os.path.join(parent_dir, "multiband")
        for m,tile_id in enumerate(tile_dict["ids"]):
            filename = "multiband_" + str(counter) + "_" + str(m)
            # schedule to download multiband
            task = asyncio.create_task(
                async_download_tile(
                    session,
                    polygon,
                    tile_id,
                    multiband_filepath,
                    counter,
                    img_count,
                    filename=filename,
                    filePerBand=False,
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
                    img_count,
                    filename=None,
                    filePerBand=True,
                )
            )
            tasks.append(task)
            img_count += 1

    await tqdm.asyncio.tqdm.gather(*tasks, position=0, desc=f"All Downloads")


async def async_download_tiles(tiles_info: List[dict], sum_imgs: int):
    async with aiohttp.ClientSession() as session:
        await fetch_all(session, tiles_info, sum_imgs)


def run_async_download(tiles_info: List[dict]):
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # apply a nested loop to jupyter's event loop for async downloading
    # nest_asyncio.apply()
    # get nested running loop and wait for async downloads to complete
    loop = asyncio.get_running_loop()
    result = loop.run_until_complete(async_download_tiles(tiles_info))


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
            "filepath": roi_path + os.sep + "tile" + str(counter),
        }
        img_dicts.append(image_dict)
        print(f"\n Images available for tile {counter} : {len(im_list)}")
        sum_imgs += len(im_list)

    print(f"\nTotal Images available across all tiles {sum_imgs}")
    return img_dicts, sum_imgs


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
roi_path = os.path.abspath(os.getcwd() + os.sep + "ROIs" + os.sep + "ROI13")
# @todo remove FOR TESTING ONLY
# if os.path.exists(roi_path):
#     shutil.rmtree(roi_path)
# create directory to hold all multiband files
multiband_path = os.path.join(roi_path, "multiband")
create_dir(multiband_path, raise_error=False)


# get list of tiles info needed to downloaded all tiles
tiles_info, sum_imgs = get_tiles_info(tile_coords, dates, roi_path, gee_collection)
print(f"Total Images to Download: {sum_imgs*2}")
# make directories for all tiles within ROI
mk_filepaths(tiles_info)
# download all the tiles within the ROI
# run_async_download(tiles_info)
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# download the tiles that compose the ROI in parallel
asyncio.run(async_download_tiles(tiles_info, sum_imgs))
# unzip all the downloaded data
unzip_data(roi_path)

# copy multiband tifs for each tile into a single directory named 'multiband'
# all multiband files are already downloaded to multiband directory
# copy_multiband_tifs(roi_path, multiband_path)

merge_tifs(multiband_path=multiband_path, roi_path=roi_path)

end = perf_counter()
total_time = end - start
with open("multithreading_timer.txt", "a") as f:
    f.write(f"\nEnd Time {end}\n Total Time: {total_time}")
