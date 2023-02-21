import os
import time
import json
import math
from typing import List
import platform
import logging
import os, json, shutil
from glob import glob
import concurrent.futures
from datetime import datetime

from src.seg2map import exceptions
from src.seg2map import common

import asyncio
import nest_asyncio
import aiohttp
import tqdm
import tqdm.auto
import tqdm.asyncio
import ee


logger = logging.getLogger(__name__)


# GEE allows for at least 20 concurrent requests at once


async def download_file(session, url, save_location):
    retries = 3  # number of times to retry download
    for i in range(retries):
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    print(f"An error occurred while downloading.{response}")
                    logger.error(f"An error occurred while downloading.{response}")
                    print(response.status)
                    return
                with open(save_location, "wb") as f:
                    async for chunk in response.content.iter_chunked(1024):
                        if not chunk:
                            break
                        f.write(chunk)
                break  # break out of retry loop if download is successful
        except asyncio.exceptions.TimeoutError as e:
            logger.error(e)
            logger.error(f"An error occurred while downloading {save_location}.{e}")
            print(
                f"Timeout error occurred for {url}. Retrying with new session in 1 second... ({i + 1}/{retries})"
            )
            await asyncio.sleep(1)
            async with aiohttp.ClientSession() as new_session:
                return await download_file(new_session, url, save_location)
        except Exception as e:
            logger.error(e)
            logger.error(
                f"Download failed for {save_location} {url}. Retrying in 1 second... ({i + 1}/{retries})"
            )
            print(
                f"Download failed for {url}. Retrying in 1 second... ({i + 1}/{retries})"
            )
            await asyncio.sleep(1)
    else:
        logger.error(f"Download failed for {save_location} {url}.")
        print(f"Download failed for {url}.")
        return


async def create_session():
    return aiohttp.ClientSession()


async def download_group(session, group, semaphore):
    coroutines = []
    logger.info(f"group: {group}")
    for tile_number, tile in enumerate(group):
        polygon = tile["polygon"]
        filepath = os.path.abspath(tile["filepath"])
        filenames = {
            "multiband": "multiband" + str(tile_number),
            "singleband": os.path.basename(filepath),
        }
        for tile_id in tile["ids"]:
            logger.info(f"tile_id: {tile_id}")
            file_id = tile_id.replace("/", "_")
            filename = filenames["multiband"] + "_" + file_id
            save_location = os.path.join(filepath, filename.replace("/", "_") + ".zip")
            logger.info(f"save_location: {save_location}")
            coroutines.append(
                async_download_tile(
                    session,
                    polygon,
                    tile_id,
                    save_location,
                    filename,
                    filePerBand=False,
                    semaphore=semaphore,
                )
            )

    # year_name=os.path.basename(group[0]['filepath'])
    # await tqdm.asyncio.tqdm.gather(*coroutines, leave=False, desc=f"Downloading {year_name}")
    await asyncio.gather(*coroutines)

    logger.info(f"Files downloaded to {group[0]['filepath']}")
    common.unzip_dir(group[0]["filepath"])
    common.delete_empty_dirs(group[0]["filepath"])


# async def download_group(session, urls, save_folder, semaphore):
#     coroutines = []
#     for url in urls:
#         filename = os.path.basename(url)
#         save_location = os.path.join(save_folder, filename)
#         async with semaphore:
#             coroutines.append(download_file(session, url, save_location))
#     await asyncio.gather(*coroutines)

# async def download_groups(url_groups, save_folder):
#     async with aiohttp.ClientSession() as session:
#         semaphore = asyncio.Semaphore(10)
#         coroutines = [download_group(session, urls, save_folder, semaphore) for urls in url_groups]
#         await asyncio.gather(*coroutines)

# Download the information for each year
# async def download_groups(groups):
#     async with aiohttp.ClientSession() as session:
#         semaphore = asyncio.Semaphore(10)
#         logger.info(f"group: {groups}")
#         for key,group in groups.items():
#             logger.info(f"key: {key} group: {group}")
#             if len(group) > 0:
#                 await download_group(session, group, semaphore)
#                 logger.info(f"Files downloaded to {group[0]['filepath']} for group: {key}")
#                 common.unzip_dir(group[0]['filepath'])
#                 common.delete_empty_dirs(group[0]['filepath'])
#             else:
#                 print(f"No tiles available to download for year: {key}")


# Download the information for each year
async def download_groups(groups, semaphore: asyncio.Semaphore):
    coroutines = []
    async with aiohttp.ClientSession() as session:
        logger.info(f"group: {groups}")
        for key, group in groups.items():
            logger.info(f"key: {key} group: {group}")
            if len(group) > 0:
                coroutines.append(download_group(session, group, semaphore))
            else:
                print(f"No tiles available to download for year: {key}")
                logger.warning(f"No tiles available to download for year: {key}")

        # await asyncio.gather(*coroutines)
        await tqdm.asyncio.tqdm.gather(
            *coroutines, position=1, leave=False, desc=f"Downloading years"
        )


# async def download_groups(groups, save_folder):
#     async with aiohttp.ClientSession() as session:
#         semaphore = asyncio.Semaphore(10)
#         for group in groups:
#             await download_group(session, group, save_folder, semaphore)
#             await cleanup(save_folder)


# Download the information for each ROI
# async def download_ROIs(ROI_tiles:dict={}):
#     for ROI_info in ROI_tiles.values():
#         logger.info(f"ROI_info: {ROI_info}")
#         await download_groups(ROI_info)


async def download_ROIs(ROI_tiles: dict = {}):
    tasks = []
    semaphore = asyncio.Semaphore(15)
    for ROI_info in ROI_tiles.values():
        tasks.append(download_groups(ROI_info, semaphore))
    # await asyncio.gather(*tasks)
    await tqdm.asyncio.tqdm.gather(
        *tasks, position=0, leave=False, desc=f"Downloading ROIs"
    )


# async def download_ROIs(ROI_tiles:dict={}):
#     tasks = []
#     for ROI_info in ROI_tiles.values():
#         task =  asyncio.create_task(download_groups(ROI_info))
#         tasks.append(task)
#     await asyncio.gather(tasks)


async def async_download_tile(
    session: aiohttp.ClientSession,
    polygon: List[set],
    tile_id: str,
    filepath: str,
    filename: str,
    filePerBand: bool,
    semaphore: asyncio.Semaphore,
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
    semaphore:asyncio.Semaphore : Limits number of concurrent requests
    Returns:
    None
    """
    # Semaphore limits number of concurrent requests
    async with semaphore:
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
            # create download url using id
            url = ee.data.makeDownloadUrl(download_id)
            await download_file(session, url, filepath)
        except Exception as e:
            logger.error(e)
            raise e


# async def create_group(tiles_info: List[dict], download_bands: str,session) -> None:
#         # creates task for each tile to be downloaded and waits for tasks to complete
#         tasks = []
#         for counter, tile_dict in enumerate(tiles_info):
#             polygon = tile_dict["polygon"]
#             filepath = os.path.abspath(tile_dict["filepath"])
#             filenames = {
#                 "multiband": "multiband" + str(counter),
#                 "singleband": os.path.basename(filepath),
#             }
#             for tile_id in tile_dict["ids"]:
#                 logger.info(f"tile_id: {tile_id}")
#                 file_id = tile_id.replace("/", "_")
#                 logger.info(f"year_filepath: {filepath}")
#                 tasks.extend(
#                     create_tasks(
#                         session,
#                         polygon,
#                         tile_id,
#                         filepath,
#                         filepath,
#                         filenames,
#                         file_id,
#                         download_bands,
#                     )
#                 )
#         # show a progress bar of all the requests in progress
#         await tqdm.asyncio.tqdm.gather(*tasks, position=0, desc=f"All Downloads")
#         common.unzip_data(os.path.dirname(filepath))
#         # delete any directories that were empty
#         common.delete_empty_dirs(os.path.dirname(filepath))


def run_async_function(async_callback, **kwargs) -> None:
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # apply a nested loop to jupyter's event loop for async downloading
    nest_asyncio.apply()
    # get nested running loop and wait for async downloads to complete
    loop = asyncio.get_running_loop()
    result = loop.run_until_complete(async_callback(**kwargs))
    logger.info(f"result: {result}")
    return result
