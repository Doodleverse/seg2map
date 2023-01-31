import json
import math
import logging
import os, json, shutil
from glob import glob
import concurrent.futures

from src.seg2map import exceptions
from src.seg2map import common

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

# Initialize a semaphore object with a limit of 10
# GEE allows for 10 concurrent requests at once
limit = asyncio.Semaphore(11)


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


def remove_zip(path) -> None:
    # Get a list of all the zipped files in the directory
    zipped_files = [
        os.path.join(path, f) for f in os.listdir(path) if f.endswith(".zip")
    ]
    # Remove each zip file
    for zipped_file in zipped_files:
        os.remove(zipped_file)


def unzip(path) -> None:
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


def get_subdirs(parent_dir: str):
    # Get a list of all the subdirectories in the parent directory
    subdirectories = []
    for root, dirs, files in os.walk(parent_dir):
        for d in dirs:
            subdirectories.append(os.path.join(root, d))
    return subdirectories


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


async def async_download_tile(
    session: aiohttp.ClientSession,
    polygon: List[set],
    tile_id: str,
    filepath: str,
    filename: str,
    filePerBand: bool,
) -> None:
    """
    Download a single tile of an Earth Engine image and save it to a zip directory.

    This function uses the Earth Engine API to crop the image to a specified polygon and download it to a zip directory with the specified filename. The number of concurrent downloads is limited to 10.

    Parameters:

    session (aiohttp.ClientSession): An instance of aiohttp session to make the download request.
    polygon (List[set]): A list of latitude and longitude coordinates that define the region to crop the image to.
    tile_id (str): The ID of the Earth Engine image to download.
    filepath (str): The path of the directory to save the downloaded zip file to.
    filename (str): The name of the zip file to be saved.
    filePerBand (bool): Whether to save each band of the image in a separate file or as a single file.
    Returns:
    None
    """
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
    # Check if path to ROI directory exists
    if not os.path.exists(roi_path):
        raise FileNotFoundError(f"{roi_path} not found.")
    # Check if path to multiband exists
    if not os.path.exists(multiband_path):
        raise FileNotFoundError(f"{multiband_path} not found.")
    try:

        # Create a list of tif files in multiband_path
        tif_files = glob(os.path.join(multiband_path, "*.tif"))
        if not tif_files:
            raise FileNotFoundError(f"No tif files found in {multiband_path}.")

        vrt_path = os.path.join(roi_path, "merged_multispectral.vrt")
        logger.info(f"vrt_path: {vrt_path}")

        ## create vrt(virtual world format) file
        vrtoptions = gdal.BuildVRTOptions(
            resampleAlg="average", srcNodata=0, VRTNodata=0
        )
        # creates a virtual world file using all the tifs and overwrites any pre-existing .vrt
        virtual_dataset = gdal.BuildVRT(vrt_path, tif_files, options=vrtoptions)
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
    """
    Copy multiband TIF files from a source folder to a destination folder.

    This function uses the glob module to search for multiband TIF files in the subfolders of a source folder, and then uses the shutil module to copy each file to a destination folder. The name of the file in the destination folder is set to the name of the file in the source folder.

    Parameters:

    roi_path (str): The path of the source folder to search for multiband TIF files.
    multiband_path (str): The path of the destination folder to copy the multiband TIF files to.
    Returns:
    None"""
    filepaths = [tile_info["filepath"] for tile_info in tiles_info]
    for filepath in filepaths:
        create_dir(filepath, raise_error=False)


def create_tasks(
    session: aiohttp.ClientSession,
    polygon: List[tuple],
    tile_id: str,
    filepath: str,
    multiband_filepath: str,
    filenames: dict,
    file_id: str,
    download_bands: str,
) -> list:
    """

    creates a list of tasks that are used to download the data.

    Parameters
    ----------
    session : aiohttp.ClientSession
        The aiohttp.ClientSession object that handles the connection with the
        api.
    polygon : List[tuple] coordinates of the polygon in lat/lon
    ex: [(1,2),(3,4)(4,3),(3,3),(1,2)]
    tile_id : str
        GEE id of the tile
        ex: 'USDA/NAIP/DOQQ/m_4012407_se_10_1_20100612'
    filepath : str
        full path to tile directory that data will be saved to
    multiband_filepath : str
        The path to a directory will save multiband files
    filenames : dict
        A dictionary of filenames:
        'singleband': name of singleband files
        'multiband': name of multiband files
    file_id : str
        name that file will be saved as based on tile_id
    download_bands : str
        type of imagery to download
        must be one of the following strings "multiband","singleband", or "both"

    Returns
    -------
    tasks : list
        A list of tasks that are used to download the data.

    """
    tasks = []
    if download_bands == "multiband" or download_bands == "both":
        task = asyncio.create_task(
            async_download_tile(
                session,
                polygon,
                tile_id,
                multiband_filepath,
                filename=filenames["multiband"] + "_" + file_id,
                filePerBand=False,
            )
        )
        tasks.append(task)
    if download_bands == "singleband" or download_bands == "both":
        task = asyncio.create_task(
            async_download_tile(
                session,
                polygon,
                tile_id,
                filepath,
                filename=filenames["singleband"] + "_" + file_id,
                filePerBand=False,
            )
        )
        tasks.append(task)
    return tasks


async def async_download_tiles(tiles_info: List[dict], download_bands: str) -> None:
    """
        Downloads all tiles asynchronously and displays a tqdm for the download progress.
        downloads the tiles in separate directories depending on whether single band or multiband
        was requested
    Args:
        tiles_info (List[dict]): list information for each tile
                each list entry contains a dictionary
                {
                    polygon (list[tuple]): coordinates of the polygon in lat/lon
                        ex: [(1,2),(3,4)(4,3),(3,3),(1,2)]
                    filepath (str): full path to tile directory
                        ex: C:/users/seg2map/data/sitename/tile0
                    ids: (list[str]): GEE ids of each file
                        ex: ['USDA/NAIP/DOQQ/m_4012407_se_10_1_20100612']

                }
        download_bands (str): type of imagery to download
            must be one of the following strings "multiband","singleband", or "both
    """
    async with aiohttp.ClientSession() as session:
        # creates task for each tile to be downloaded and waits for tasks to complete
        tasks = []
        for counter, tile_dict in enumerate(tiles_info):

            polygon = tile_dict["polygon"]
            filepath = os.path.abspath(tile_dict["filepath"])
            parent_dir = os.path.dirname(filepath)
            multiband_filepath = os.path.join(parent_dir, "multiband")
            filenames = {
                "multiband": "multiband" + str(counter),
                "singleband": os.path.basename(filepath),
            }
            for tile_id in tile_dict["ids"]:
                file_id = tile_id.replace("/", "_")
                year_str = file_id.split("_")[-1][:4]
                logger.info(f"year_str: {year_str}")
                # full path to year directory within multiband dir eg. ./multiband/2012
                year_filepath = os.path.join(multiband_filepath, year_str)
                logger.info(f"year_filepath: {year_filepath}")
                tasks.extend(
                    create_tasks(
                        session,
                        polygon,
                        tile_id,
                        filepath,
                        year_filepath,
                        filenames,
                        file_id,
                        download_bands,
                    )
                )
        # show a progress bar of all the requests in progress
        await tqdm.asyncio.tqdm.gather(*tasks, position=0, desc=f"All Downloads")


# call asyncio to run download_ROIs
def run_async_download(
    download_path: str,
    roi_gdf: gpd.GeoDataFrame,
    ids: List[str],
    dates: Tuple[str],
    download_bands: str,
) -> None:
    """creates a nested loop that's used to asynchronously download imagery and waits for all the imagery to download
    Args:
        download_path (str): full path to directory to download imagery to
        roi_gdf (gpd.GeoDataFrame): geodataframe of ROIs on the map
        ids (List[str]): ids of ROIs to download imagery for
        dates (Tuple[str]): start and end dates
        download_bands (str): type of imagery to download
            must be one of the following strings "multiband","singleband", or "both"
    """
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # apply a nested loop to jupyter's event loop for async downloading
    nest_asyncio.apply()
    # get nested running loop and wait for async downloads to complete
    loop = asyncio.get_running_loop()
    loop.run_until_complete(
        download_rois(download_path, roi_gdf, ids, dates, download_bands)
    )


def get_tiles_info(
    tile_coords: list, dates: Tuple[str], roi_path: str, gee_collection: str
) -> List[dict]:
    """
    Get information about images within the specified tile coordinates, date range, and image collection.
    The information includes the image IDs, file path, and polygon geometry of each tile.

    Parameters:
    tile_coords (List[List[Tuple[float]]]): A list of tile coordinates, where each tile coordinate is a list of
                                            (latitude, longitude) tuples that define the polygon of the tile.
    dates (Tuple[str]): A tuple of two strings representing the start and end dates for the image collection.
    roi_path (str): The path to the directory where the images will be saved.
    gee_collection (str): The name of the image collection on Google Earth Engine.

    Returns:
    List[dict]: A list of dictionaries, where each dictionary contains information about a single tile, including the
                polygon geometry, the IDs of the images within the tile, and the file path to the directory where the
                images will be saved.
    """
    sum_imgs = 0
    image_ids = []
    image_dicts = []
    for counter, tile in enumerate(tile_coords):
        collection = ee.ImageCollection(gee_collection)
        polygon = ee.Geometry.Polygon(tile)
        # Filter the collection to get only the images within the tile and date range
        filtered_collection = (
            collection.filterBounds(polygon)
            .filterDate(*dates)
            .sort("system:time_start", True)
        )
        # Get a list of all the image names in the filtered collection
        image_list = filtered_collection.getInfo().get("features")
        ids = [obj["id"] for obj in image_list]
        image_ids.extend(ids)
        # Create a dictionary for each tile with the information about the images to be downloaded
        image_dict = {
            "polygon": tile,
            "ids": ids,
            "filepath": os.path.join(roi_path, f"tile{str(counter)}"),
        }
        image_dicts.append(image_dict)
        logger.info(f"\n Images available for tile {counter} : {len(image_list)}")
        sum_imgs += len(image_list)

    logger.info(f"\nTotal Images available across all tiles {sum_imgs}")
    return image_dicts, sum_imgs


def get_tile_coords(num_splitters: int, roi_gdf: gpd.GeoDataFrame) -> list[list[list]]:
    """
    Given the number of splitters and a GeoDataFrame,Splits an ROI geodataframe into tiles of 1km^2 area (or less),
    and returns a list of lists of tile coordinates.

    Args:
        num_splitters (int): The number of splitters to divide the ROI into.
        gpd_data (gpd.GeoDataFrame): The GeoDataFrame containing the ROI geometry.

    Returns:
        list[list[list[float]]]: A list of lists of tile coordinates, where each inner list represents the coordinates of one tile.
        The tile coordinates are in [lat,lon] format.
    """
    if num_splitters == 0:
        split_polygon = Polygon(roi_gdf["geometry"].iloc[0])
        tile_coords = [list(split_polygon.exterior.coords)]
    elif num_splitters > 0:
        # split ROI into rectangles of 1km^2 area (or less)
        split_polygon = splitPolygon(roi_gdf, num_splitters)
        tile_coords = [list(part.exterior.coords) for part in split_polygon.geoms]
    return tile_coords


async def download_ROI(
    download_path: str,
    roi_gdf: gpd.GeoDataFrame,
    roi_id: str,
    dates: Tuple[str],
    download_bands: str,
) -> None:
    """
    Download the imagery data for a given region of interest (ROI) within specified dates. The ROI is split into smaller rectangles to minimize the amount of data that needs to be downloaded. The downloaded data is unzipped and merged into a single multiband image.

    Parameters:
        download_path (str): The path where the downloaded data should be saved.
        roi_gdf (gpd.GeoDataFrame): The geographical data for the ROI.
        roi_id (str): A string identifier for the ROI.
        dates (Tuple[str]): A tuple of two strings representing the start and end date of the imagery data to be downloaded.
        download_bands (str): A string indicating which bands should be downloaded, either "multiband", "singleband", or "both".

    Returns:
        None
    """
    gee_collection = "USDA/NAIP/DOQQ"

    # get number of splitters need to split ROI into rectangles of 1km^2 area (or less)
    num_splitters = get_num_splitters(roi_gdf)
    logger.info("Splitting ROI into {num_splitters}x{num_splitters} tiles")

    # split ROI into rectangles of 1km^2 area (or less)
    tile_coords = get_tile_coords(num_splitters, roi_gdf)
    logger.info(f"tile_coords: {tile_coords}")

    # name of ROI folder to contain all downloaded data
    roi_name = f"ID_{roi_id}_dates_{dates[0]}_to_{dates[1]}"
    roi_path = os.path.join(download_path, roi_name)
    # create directory to hold all multiband files
    multiband_path = os.path.join(roi_path, "multiband")
    create_dir(multiband_path, raise_error=False)

    # create subdirectories for each year
    start_date = dates[0].split("-")[0]
    end_date = dates[1].split("-")[0]
    logger.info(f"start_date : {start_date } end_date : {end_date }")
    common.create_year_directories(int(start_date), int(end_date), multiband_path)

    # Get list of tile info needed for download
    tiles_info, sum_imgs = get_tiles_info(tile_coords, dates, roi_path, gee_collection)

    if sum_imgs == 0:
        raise exceptions.No_Images_Available(
            f"No images found within these dates : {dates}"
        )
    # if both single band and multiband imagery will be downloaded then double number images
    sum_imgs *= 2 if download_bands == "both" else 1
    print(f"Total Images to Download: {sum_imgs}")
    logger.info(f"Total Images to Download: {sum_imgs}")

    # make directories for all tiles within ROI
    mk_filepaths(tiles_info)
    # download ROI tiles concurrently
    await async_download_tiles(tiles_info, download_bands)
    # Download tiles and unzip data
    unzip_data(roi_path)
    # delete any directories that were empty
    common.delete_empty_dirs(multiband_path)
    # create multispectral tif for each year only if multiband imagery was downloaded
    if download_bands != "singleband":
        for subdir in os.scandir(multiband_path):
            merge_tifs(multiband_path=subdir.path, roi_path=subdir.path)


async def download_rois(
    download_path: str,
    roi_gdf: gpd.GeoDataFrame,
    ids: List[str],
    dates: Tuple[str],
    download_bands: str,
) -> None:
    no_imgs_counter = 0
    for roi_id in ids:
        # download selected ROI
        try:
            gpd_data = roi_gdf[roi_gdf["id"] == roi_id]
            await download_ROI(download_path, gpd_data, roi_id, dates, download_bands)
        except exceptions.No_Images_Available as error:
            no_imgs_counter += 1
    # if every ROI has no images available raise error
    if no_imgs_counter == len(ids):
        raise exceptions.No_Images_Available(
            f"No images found within these dates : {dates}"
        )
