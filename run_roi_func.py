import json
import math
import os, json, shutil
from glob import glob
from typing import List, Tuple
import platform

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
from datetime import datetime

from time import perf_counter

start = perf_counter()
with open("async_timer.txt", "w") as f:
    f.write(f"Start Time {start}")


# pass in ROI, ROI's id, dates, sitename
# -----------------------------------
sitename = "2010"
# empty no data for these dates
# dates=('2018-12-01', '2019-03-01')
# data for 2009,2010
dates = ("2010-01-01", "2010-12-31")

# -----------------------------------
# create sitename directory
site_path = os.path.abspath(os.getcwd() + os.sep + sitename)
if os.path.exists(site_path):
    shutil.rmtree(site_path)
create_dir(site_path)

# loop through each ROI
roi_id = "1"
roi_name = "ID_" + roi_id + "_dates_" + dates[0] + "_to_" + dates[1]
# isolate a single ROI's geojson
filename = (
    r"C:\1_USGS\5_Doodleverse\1_Seg2Map_fork\seg2map\src\seg2map\NAIP\map14.geojson"
)
with open(filename, "r") as f:
    gpd_data = gpd.read_file(f)

# get number of splitters need to split ROI into rectangles of 1km^2 area (or less)
num_splitters = get_num_splitters(gpd_data)
print(num_splitters)
# split ROI into rectangles of 1km^2 area (or less)
split_polygon = splitPolygon(gpd_data, num_splitters)

# create path to roi directory which will hold sub directories of tiles

roi_path = os.path.abspath(site_path + os.sep + roi_name)
print(f"roi_path: {roi_path}")
create_dir(roi_path)
# create multiband directory
multiband_path = os.path.join(roi_path, "multiband")
print(f"multiband_path: {multiband_path}")
create_dir(multiband_path)

gee_collection = "USDA/NAIP/DOQQ"
ee.Initialize()


asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# download the tiles that compose the ROI in parallel
asyncio.run(async_download_tiles(split_polygon, roi_path, gee_collection, dates))


# DO THESE STEPS AFTER ALL TILES FOR A GIVEN ROI HAVE BEEN DOWNLOADED
# copy multiband tifs for each tile into a single directory named 'multiband'
copy_multiband_tifs(roi_path, multiband_path)
# run_async_download(split_polygon,site_path,gee_collection,dates)
merge_tifs(tifs_path=multiband_path, roi_path=roi_path)

end = perf_counter()
total_time = end - start
with open("async_timer.txt", "a") as f:
    f.write(f"\nEnd Time {end}\n Total Time: {total_time}")
