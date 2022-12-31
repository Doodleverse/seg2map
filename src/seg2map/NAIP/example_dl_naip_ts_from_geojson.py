# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2022, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# For a list of polygons in a geojson file

# 1. download all the 0.5-m NAIP imagery. Bands downloaded separately and as a multiband geotiff
# 1a. If the imagery is larger than an allowable GEE data packet (33MB), region is split into nx x ny regions

# (you need to run earthengine authenticate in the cmd first)

import geemap, ee
import os, json, shutil
import numpy as np
from glob import glob
from area import area
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import split
from osgeo import gdal

###=================================================
############### USER INPUTS #########################
###=================================================
## todo: make a config file containing these settings, and read in config file

MAXCLOUD = 5  # percentage maximum cloud cover tolerated

OUT_RES_M = 0.5  # output raster spatial footprint in metres

nx, ny = 2, 1  # number of columns and rows to split

# site = 'beaches' #name of the site
# site = 'hidden_beach' #name of the site
site = "hidden_beachs"  # name of the site
# name of file containing
# roifile = 'example_singleROI.geojson'
roifile = 'hidden_beach.geojson'
# roifile = "multi_sites_tests.geojson"
## years of data collection (list of strings)
# years = ['2006', '2009','2011','2013','2015','2017']
# years = ['2017'] # not available with example
# years = ['2015']
# years = ["2006", "2009", "2011", "2013", "2015", "2017"]
years=["2009"]
###=================================================
############### DOWNLOAD NAIP TILES USING GEEMAP ###
###=================================================

# initialize Earth Engine
# ee.Authenticate()
ee.Initialize()

## todo: add checks to ensure geojson is valid
with open(roifile) as f:
    json_data = json.load(f)

features = json_data["features"]  # [0]

gee_collection = "USDA/NAIP/DOQQ"

# cycle through each feature and each year
# make polygon splits
# for each split, download the tiles
# at the end of the feature/year, copy multiband data over to a separate folder
# then create a VRT, geoTIFF, and JPEG with WLD file
for featurenumber, feature in enumerate(features):

    for year in years:
        print("Working on {}".format(year))

        start_date = year + "-01-01"
        end_date = year + "-12-31"

        try:
            os.mkdir(site)
        except:
            pass

        try:
            os.mkdir(site + os.sep + "feature" + str(featurenumber))
        except:
            pass

        try:
            os.mkdir(site + os.sep + "feature" + str(featurenumber) + os.sep + year)
        except:
            pass

        coordinates = feature["geometry"]["coordinates"][0]

        lng, lat = np.mean(coordinates, axis=0)

        area_sqkm = area(feature["geometry"]) / 1e6

        collection = ee.ImageCollection(gee_collection)

        polygon = Polygon([tuple(c) for c in coordinates])

        minx, miny, maxx, maxy = polygon.bounds

        dx = (maxx - minx) / nx  # width of a small part
        dy = (maxy - miny) / ny  # height of a small part
        horizontal_splitters = [
            LineString([(minx, miny + i * dy), (maxx, miny + i * dy)])
            for i in range(ny)
        ]
        vertical_splitters = [
            LineString([(minx + i * dx, miny), (minx + i * dx, maxy)])
            for i in range(nx)
        ]
        splitters = horizontal_splitters + vertical_splitters

        result = polygon
        for splitter in splitters:
            result = MultiPolygon(split(result, splitter))
        parts = [list(part.exterior.coords) for part in result.geoms]

        print("Number of individual tiles: {}".format(len(parts)))

        counter = 1
        for part in parts:
            print("Working on part {} of {}".format(counter, len(parts)))
            try:
                os.mkdir(
                    site
                    + os.sep
                    + "feature"
                    + str(featurenumber)
                    + os.sep
                    + year
                    + os.sep
                    + "chunk"
                    + str(counter)
                )
            except:
                pass

            collection = ee.ImageCollection(gee_collection)

            polys = ee.Geometry.Polygon(part)

            centroid = polys.centroid()
            lng, lat = centroid.getInfo()["coordinates"]

            collection = collection.filterBounds(polys)
            collection = collection.filterDate(start_date, end_date).sort(
                "system:time_start", True
            )
            count = collection.size().getInfo()
            # print("Number of cloudy scenes: ", count)
            img_lst = collection.toList(1000)

            N = []
            for i in range(0, count):
                image = ee.Image(img_lst.get(i))
                name = image.get("system:index").getInfo()
                N.append(name)
                print(name)

            for n in N:
                # Export each band as one image
                image = ee.Image("USDA/NAIP/DOQQ/" + n)
                geemap.ee_export_image(
                    image,
                    os.getcwd()
                    + os.sep
                    + site
                    + os.sep
                    + "feature"
                    + str(featurenumber)
                    + os.sep
                    + year
                    + os.sep
                    + "chunk"
                    + str(counter)
                    + os.sep
                    + "chunk"
                    + str(counter)
                    + "_"
                    + n
                    + ".tif",
                    scale=OUT_RES_M,
                    region=polys,
                    file_per_band=True,
                    crs="EPSG:4326",
                )

                geemap.ee_export_image(
                    image,
                    os.getcwd()
                    + os.sep
                    + site
                    + os.sep
                    + "feature"
                    + str(featurenumber)
                    + os.sep
                    + year
                    + os.sep
                    + "chunk"
                    + str(counter)
                    + os.sep
                    + "chunk"
                    + str(counter)
                    + "_"
                    + n
                    + "_multiband.tif",
                    scale=OUT_RES_M,
                    region=polys,
                    file_per_band=False,
                    crs="EPSG:4326",
                )

            counter += 1

        # move multiband files into own directory

        outdirec = (
            os.getcwd()
            + os.sep
            + site
            + os.sep
            + "feature"
            + str(featurenumber)
            + os.sep
            + year
            + os.sep
            + "multiband"
        )
        try:
            os.mkdir(outdirec)
        except:
            pass

        for folder in glob(
            os.getcwd()
            + os.sep
            + site
            + os.sep
            + "feature"
            + str(featurenumber)
            + os.sep
            + year
            + os.sep
            + "chunk*",
            recursive=True,
        ):
            files = glob(folder + os.sep + "*multiband.tif")
            [
                shutil.copyfile(file, outdirec + os.sep + file.split(os.sep)[-1])
                for file in files
            ]

        # ## run gdal to merge into big tiff
        try:
            ## create vrt
            vrtoptions = gdal.BuildVRTOptions(
                resampleAlg="average", srcNodata=0, VRTNodata=0
            )
            files = glob(outdirec + os.sep + "*multiband.tif")
            outfile = (
                os.getcwd()
                + os.sep
                + site
                + os.sep
                + "feature"
                + str(featurenumber)
                + os.sep
                + year
                + os.sep
                + "merged_multispectral.vrt"
            )
            ds = gdal.BuildVRT(outfile, files, options=vrtoptions)
            ds.FlushCache()
            ds = None

            # create geotiff
            ds = gdal.Translate(
                outfile.replace(".vrt", ".tif"),
                creationOptions=["COMPRESS=LZW", "TILED=YES"],
                srcDS=outfile,
            )
            ds.FlushCache()
            ds = None

            # create jpeg
            ds = gdal.Translate(
                outfile.replace(".vrt", ".jpg"),
                creationOptions=["WORLDFILE=YES", "QUALITY=100"],
                srcDS=outfile.replace(".vrt", ".tif"),
            )
            ds.FlushCache()
            ds = None

        except:
            print("Data not available")
