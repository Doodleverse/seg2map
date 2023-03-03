
import os
from osgeo import gdal
from typing import List

def build_vrt(vrt: str, files: List[str], resample_name: str) -> None:
    """builds .vrt file which will hold information needed for overlay
    Args:
        vrt (:obj:`string`): name of vrt file, which will be created
        files (:obj:`list`): list of file names for merging
        resample_name (:obj:`string`): name of resampling method
    """

    options = gdal.BuildVRTOptions(srcNodata=0)
    gdal.BuildVRT(destName=vrt, srcDSOrSrcDSTab=files, options=options)
    add_pixel_fn(vrt, resample_name)


def add_pixel_fn(filename: str, resample_name: str) -> None:
    """inserts pixel-function into vrt file named 'filename'
    Args:
        filename (:obj:`string`): name of file, into which the function will be inserted
        resample_name (:obj:`string`): name of resampling method
    """

    header = """  <VRTRasterBand dataType="Byte" band="1" subClass="VRTDerivedRasterBand">"""
    contents = """
    <PixelFunctionType>{0}</PixelFunctionType>
    <PixelFunctionLanguage>Python</PixelFunctionLanguage>
    <PixelFunctionCode><![CDATA[{1}]]>
    </PixelFunctionCode>"""

    lines = open(filename, 'r').readlines()
    lines[3] = header  # FIX ME: 3 is a hand constant
    lines.insert(4, contents.format(resample_name,
                                    get_resample(resample_name)))
    open(filename, 'w').write("".join(lines))


def get_resample(name: str) -> str:
    """retrieves code for resampling method
    Args:
        name (:obj:`string`): name of resampling method
    Returns:
        method :obj:`string`: code of resample method
    """

    methods = {
        "first":
        """
import numpy as np
def first(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    y = np.ones(in_ar[0].shape)
    for i in reversed(range(len(in_ar))):
        mask = in_ar[i] == 0
        y *= mask
        y += in_ar[i]
    np.clip(y,0,255, out=out_ar)
""",
        "last":
        """
import numpy as np
def last(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    y = np.ones(in_ar[0].shape)
    for i in range(len(in_ar)):
        mask = in_ar[i] == 0
        y *= mask
        y += in_ar[i]
    np.clip(y,0,255, out=out_ar)
""",
        "max":
        """
import numpy as np
def max(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    y = np.max(in_ar, axis=0)
    np.clip(y,0,255, out=out_ar)
""",
        "average":
        """
import numpy as np
def average(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    div = np.zeros(in_ar[0].shape)
    for i in range(len(in_ar)):
        div += (in_ar[i] != 0)
    div[div == 0] = 1
    
    y = np.sum(in_ar, axis = 0, dtype = 'uint16')
    y = y / div
    
    np.clip(y,0,255, out = out_ar)
"""}

    if name not in methods:
        raise ValueError(
            "ERROR: Unrecognized resampling method (see documentation): '{}'.".
            format(name))

    return methods[name]


## make vrt
# os.system('gdalbuildvrt -srcnodata 0 -vrtnodata 0 mosaic.vrt --optfile tifs.txt')

with open('tifs.txt', 'r') as f:
    files = f.read()

vrt_options = gdal.BuildVRTOptions(
    resampleAlg="mode", srcNodata=0, VRTNodata=0
)

# ## https://github.com/aerospaceresearch/DirectDemod/blob/Vladyslav_dev/directdemod/merger.py
build_vrt(os.getcwd()+os.sep+TEMP_VRT_FILE, files, 'max')

ds = gdal.BuildVRT('special_test.vrt', wood_files, options=vrt_options)
ds.FlushCache()
ds = None 


def merge_files(src_files: str, dest_path: str, create_jpg: bool = True) -> str:
    """Merge a list of GeoTIFF files into a single JPEG file.

    Args:
    src_files (List[str]): A list of file paths to be merged.
    dest_path (str): The path to the output JPEG file.

    Returns:
    str: The path to the output JPEG file.
    """
    # Check if path to source exists
    for file in src_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"{file} not found.")
    try:
        ## create vrt(virtual world format) file
        # Create VRT file
        vrt_options = gdal.BuildVRTOptions(
            resampleAlg="mode", srcNodata=0, VRTNodata=0
        )
        print(f"dest_path: {dest_path}")
        # creates a virtual world file using all the tifs and overwrites any pre-existing .vrt
        virtual_dataset = gdal.BuildVRT(dest_path, src_files, options=vrt_options)
        # flushing the cache causes the vrt file to be created
        virtual_dataset.FlushCache()
        # reset the dataset object
        virtual_dataset = None

        print(f"dest_path: {dest_path}")
        build_vrt(dest_path, src_files, 'max')
        print(f"after new  build_vrt dest_path: {dest_path}")

        # create geotiff (.tiff) from merged vrt file
        tif_path = dest_path.replace(".vrt", ".tif")
        virtual_dataset = gdal.Translate(
            tif_path,
            creationOptions=["COMPRESS=LZW", "TILED=YES"],
            srcDS=dest_path,
        )
        virtual_dataset.FlushCache()
        virtual_dataset = None

        if create_jpg:
            # convert .vrt to .jpg file
            virtual_dataset = gdal.Translate(
                dest_path.replace(".vrt", ".jpg"),
                creationOptions=["WORLDFILE=YES", "QUALITY=100"],
                srcDS=dest_path.replace(".vrt", ".tif"),
            )
            virtual_dataset.FlushCache()
            virtual_dataset = None

        return dest_path
    except Exception as e:
        print(e)
        raise e