import json
import math
import logging
import os, json, shutil
from glob import glob
import concurrent.futures

from src.seg2map import  exceptions

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
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import split
import ee
from osgeo import gdal


logger = logging.getLogger(__name__)


def get_num_splitters(gdf: gpd.GeoDataFrame) -> int:
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
    roi_geometry = roi_json["features"][0]["geometry"]
    # get area of entire shape as squared kilometers
    area_km2 = area(roi_geometry) / 1e6
    logger.info(f"Area: {area_km2}")
    if area_km2 <= 1:
        return 0
    # get minimum number of horizontal and vertical splitters to split area equally
    # max area per tile is 1 km^2
    num_splitters = math.ceil(math.sqrt(area_km2))
    return num_splitters


def splitPolygon(polygon: gpd.GeoDataFrame, num_splitters: int) -> MultiPolygon:
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
    horizontal_splitters = [
        LineString([(minx, miny + i * dy), (maxx, miny + i * dy)])
        for i in range(num_splitters)
    ]
    vertical_splitters = [
        LineString([(minx + i * dx, miny), (minx + i * dx, maxy)])
        for i in range(num_splitters)
    ]
    splitters = horizontal_splitters + vertical_splitters
    result = polygon["geometry"].iloc[0]
    for splitter in splitters:
        result = MultiPolygon(split(result, splitter))
    return result


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
    remove_zip_files(subdirs)


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
    filename: str,
    filePerBand: bool,
):
    # No more than 10 concurrent workers will be able to make
    # get request at the same time.
    async with limit:
        OUT_RES_M = 0.5  # output raster spatial footprint in metres
        image_ee = ee.Image(tile_id)
        # crop and download
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
            # file path to zip directory that will be downloaded
            fp_zip = os.path.join(filepath, filename.replace("/", "_") + ".zip")

            chunk_size: int = 2048
            # create download url using id
            url = ee.data.makeDownloadUrl(download_id)

            # When workers hit the limit, they'll wait for a second
            # before making more requests.
            if limit.locked():
                logger.info("Concurrency limit reached, waiting ...")
                await asyncio.sleep(1)

            # zip directory images will be downloaded to
            async with session.get(url, timeout=300, raise_for_status=True) as r:
                if r.status != 200:
                    print("An error occurred while downloading.{r}")
                    logger.error("An error occurred while downloading.{r}")
                    print(r.status)
                    return
                with open(fp_zip, "wb") as fd:
                    async for chunk in r.content.iter_chunked(chunk_size):
                        fd.write(chunk)
        except Exception as e:
            logger.error(e)
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
        logger.info(f"outfile for vrt: {vrt_path}")
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
        logger.error(e)
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
        for tile_id in tile_dict["ids"]:
            file_id = tile_id.replace("/", "_")
            multi_filename = "multiband" + str(counter) + "_" + file_id
            filename = os.path.basename(filepath) + "_" + file_id
            # schedule to download multiband
            task = asyncio.create_task(
                async_download_tile(
                    session,
                    polygon,
                    tile_id,
                    multiband_filepath,
                    filename=multi_filename,
                    filePerBand=False,
                )
            )
            tasks.append(task)
            # schedule to download each band as a single file
            task = asyncio.create_task(
                async_download_tile(
                    session,
                    polygon,
                    tile_id,
                    filepath,
                    filename=filename,
                    filePerBand=True,
                )
            )
            tasks.append(task)
            img_count += 1
    # show a progress bar of all the requests in progress
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
        logger.info(f"\n Images available for tile {counter} : {len(im_list)}")
        sum_imgs += len(im_list)
        
    logger.info(f"\nTotal Images available across all tiles {sum_imgs}")
    return img_dicts, sum_imgs


def get_tile_coords(num_splitters: int, gpd_data):
    if num_splitters == 0:
        split_polygon = Polygon(gpd_data["geometry"].iloc[0])
        tile_coords = [list(split_polygon.exterior.coords)]
    elif num_splitters > 0:
        # split ROI into rectangles of 1km^2 area (or less)
        split_polygon = splitPolygon(gpd_data, num_splitters)
        tile_coords = [list(part.exterior.coords) for part in split_polygon.geoms]
    return tile_coords


async def download_ROI(
    download_path: str, gpd_data: gpd.GeoDataFrame, roi_id: str, dates: Tuple[str]
) -> None:
    gee_collection = "USDA/NAIP/DOQQ"
    # get number of splitters need to split ROI into rectangles of 1km^2 area (or less)
    num_splitters = get_num_splitters(gpd_data)
    logger.info("Splitting ROI into {num_splitters}x{num_splitters} tiles")
    # split ROI into rectangles of 1km^2 area (or less)
    tile_coords = get_tile_coords(num_splitters, gpd_data)
    logger.info(f"tile_coords: {tile_coords}")

    # name of ROI folder to contain all downloaded data
    roi_name = "ID_" + roi_id + "_dates_" + dates[0] + "_to_" + dates[1]
    roi_path = os.path.join(download_path, roi_name)
    # create directory to hold all multiband files
    multiband_path = os.path.join(roi_path, "multiband")
    create_dir(multiband_path, raise_error=False)
    # get list of tiles info needed to downloaded all tiles
    tiles_info, sum_imgs = get_tiles_info(tile_coords, dates, roi_path, gee_collection)
    if sum_imgs == 0:
        raise exceptions.No_Images_Available(f"No images found within these dates : {dates}")
    print(f"Total Images to Download: {sum_imgs*2}")
    logger.info(f"Total Images to Download: {sum_imgs*2}")
    # make directories for all tiles within ROI
    mk_filepaths(tiles_info)
    # download the tiles that compose the ROI in parallel
    await async_download_tiles(tiles_info, sum_imgs)
    # unzip all the downloaded data
    unzip_data(roi_path)
    # merge all multiband tifs into a single jpg
    merge_tifs(multiband_path=multiband_path, roi_path=roi_path)


async def download_rois(
    download_path: str, roi_gdf: gpd.GeoDataFrame, ids: List[str], dates: Tuple[str]
) -> None:
    no_imgs_counter=0
    for roi_id in ids:
        # download selected ROI
        try:
            gpd_data = roi_gdf[roi_gdf["id"] == roi_id]
            await download_ROI(download_path, gpd_data, roi_id, dates)
        except exceptions.No_Images_Available as error:
            no_imgs_counter+=1
    # if every ROI has no images available raise error        
    if no_imgs_counter == len(ids):
        raise exceptions.No_Images_Available(f"No images found within these dates : {dates}")
    
# call asyncio to run download_ROIs
def run_async_download(
    download_path: str, roi_gdf: gpd.GeoDataFrame, ids: List[str], dates: Tuple[str]
):
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # apply a nested loop to jupyter's event loop for async downloading
    nest_asyncio.apply()
    # get nested running loop and wait for async downloads to complete
    loop = asyncio.get_running_loop()
    result = loop.run_until_complete(download_rois(download_path, roi_gdf, ids, dates))
