import os
import re
import random
import string
from pathlib import Path
import glob
import shutil
import json
import math
from datetime import datetime
import logging
from typing import Set, Union, List
import json
import math
import logging
import os, json, shutil
from glob import glob
import concurrent.futures
from datetime import datetime
from time import perf_counter

# Internal dependencies imports
from seg2map import map_functions

from tqdm.auto import tqdm
import requests
import zipfile
from area import area
import geopandas as gpd
import numpy as np
import geojson
import matplotlib
from leafmap import check_file_path
from osgeo import gdal
from skimage.io import imsave
import time
from PIL import Image
import rasterio
from ipywidgets import ToggleButton
from ipywidgets import HBox
from ipywidgets import VBox
from ipywidgets import Layout
from ipywidgets import HTML
from ipyfilechooser import FileChooser


logger = logging.getLogger(__name__)


def time_func(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds to run.")
        return result

    return wrapper


def find_file(path: str, filename: str = "session.json"):
    # if session.json is found in main directory then session path was identified
    filepath = os.path.join(path, filename)
    if os.path.isfile(filepath):
        return filepath
    else:
        parent_directory = os.path.dirname(path)
        filepath = os.path.join(parent_directory, filename)
        if os.path.isfile(filepath):
            return filepath
        else:
            raise ValueError(
                f"File '{filename}' not found in the parent directory: {parent_directory} or path"
            )


def write_greylabel_to_png(npz_location: str) -> str:
    """
    Given the path of an .npz file containing a 'grey_label' key with an array of uint8 values,
    writes the array to a PNG file with the same name and location as the .npz file, with the extension
    changed to .png. Returns the path of the PNG file.

    Parameters:
    npz_location (str): The path of the .npz file to read from.

    Returns:
    str: The path of the written PNG file.
    """
    png_path = npz_location.replace(".npz", ".png")
    with np.load(npz_location) as data:
        dat = 1 + np.round(data["grey_label"].astype("uint8"))
    imsave(png_path, dat, check_contrast=False, compression=0)
    return png_path


def create_greylabel_pngs(full_path: str) -> List[str]:
    """
    Given a directory path, finds all .npz files in the directory, writes the 'grey_label' array of each .npz
    file to a corresponding PNG file, and returns a list of the paths of the written PNG files.

    Parameters:
    full_path (str): The path of the directory to search for .npz files.

    Returns:
    List[str]: A list of the paths of the written PNG files.
    """
    png_files = []
    npzs = sorted(glob(os.path.join(full_path, "*.npz")))
    for npz in npzs:
        png_files.append(write_greylabel_to_png(npz))
    return png_files

def validate_is_roi_directory(base_path: str) -> bool:
    """
    Validates if the given base_path is a Region of Interest (ROI) directory by checking for the presence of
    both a "multiband" subdirectory and a "config" file.

    Args:
        base_path (str): The path to the base directory to be validated.

    Returns:
        bool: True if base_path contains the "multiband" subdirectory and the "config" file, otherwise False.
    """
    
    # Flags to check for the existence of the required items
    has_mulitband = False
    has_config = False

    # Iterate over items in the directory
    for item in base_path.iterdir():
        # Check if the item is a directory and starts with "multiband"
        if item.is_dir() and item.name.startswith("multiband"):
            has_mulitband = True

        # Check if the item is a file and starts with "config"
        if item.is_file() and item.name.startswith("config"):
            has_config = True

    # Return True if both the "multiband" directory and the "config" file are found in base_path
    return has_mulitband and has_config

def get_subdirectories_with_ids(base_path: str) -> dict:
    """
    This function takes a base path as input and returns a dictionary containing
    the IDs and paths of the parent directory as well as any subdirectories that start with "ID_".
    
    Args:
        base_path (str): The base path to search for subdirectories.
        
    Returns:
        dict: A dictionary with IDs as keys and subdirectory paths as values.
    """
    base_path = Path(base_path)
    subdirs_with_ids = {}

    if base_path.exists():
        if base_path.is_dir() and base_path.name.startswith("ID_"):
            # make sure this is a valid ROI directory and not just a directory with an ID
            if validate_is_roi_directory(base_path):
                    roi_id = base_path.name.split("_")[1]
                    subdirs_with_ids[roi_id]=str(base_path)

    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith("ID_"):
            roi_id = item.name.split("_")[1]
            subdirs_with_ids[roi_id] = str(item)

    return subdirs_with_ids

def check_id_subdirectories_exist(base_path: str) -> bool:
    """
    Check if any subdirectories with names starting with "ID_" exist in the given base_path.

    Args:
        base_path (str): The absolute or relative path to the directory to look in.

    Returns:
        bool: True if any subdirectories with names starting with "ID_" are found, otherwise False.
    """
    # Check if the given base_path exists
    if not os.path.exists(base_path):
        return False

    # Convert base_path to a Path object
    base_path = Path(base_path)

    # Check if the base_path itself meets the condition
    if base_path.is_dir() and base_path.name.startswith("ID_"):
        return True

    # Iterate over the items in the base_path
    for item in base_path.iterdir():
        # Check if the item is a subdirectory and if its name starts with "ID_"
        if item.is_dir() and item.name.startswith("ID_"):
            return True

    # If no subdirectories with names starting with "ID_" are found, return False
    return False


def extract_roi_id_from_path(path):
    """
    Extracts roi_id from a directory name in a given path.

    The function assumes that the directory name is in the format "ID_{roiid}_dates_*".

    Args:
        path (str): A string representing the path to the directory containing the roi id.

    Returns:
        A string representing the extracted roi id, or None if the roi id is not found.
    """
    start_index = path.find("ID_") + len("ID_")
    end_index = path.find("_dates_")
    if start_index != -1 and end_index != -1:
        return path[start_index:end_index]
    else:
        return None


def read_text_file(file_path: str) -> List[str]:
    """
    Read the contents of a text file and return them as a list of strings.

    Args:
        file_path: The path to the text file to read.

    Returns:
        A list of strings representing the lines of text in the file. The list will not include any line ending characters
        ('\n', '\r', or '\r\n').

    Raises:
        ValueError: If the file does not exist at the specified file path.
    """
    if not os.path.isfile(file_path):
        raise ValueError(f"{os.path.basename(file_path)} did not exist at {file_path}")
    with open(file_path) as f:
        data = f.read().split("\n")
    return data


def convert_to_rgb(img_path: str) -> str:
    """
    Converts an image to RGB format and saves it to a new file.
    Compatable image types: '.jpg' and '.png'

    Parameters:
    img_path (str): The file path of the image to be converted.

    Returns:
    str: The file path of the converted image.
    """
    logger.info(f"img_path: {img_path}")
    if img_path.endswith(".jpg"):
        im = Image.open(img_path, formats=("JPEG",)).convert("RGB")
        out_path = img_path.replace(".jpg", "_RGB.jpg")
    elif img_path.endswith(".png"):
        im = Image.open(img_path, formats=("PNG",)).convert("RGB")
        out_path = img_path.replace(".png", "_RGB.png")
    im.save(out_path)
    logger.info(f"out_path: {out_path}")
    return out_path


def get_bounds(tif_path):
    """
    Gets the bounds of a GeoTIFF file.

    Parameters:
    tif_path (str): The file path of the GeoTIFF file.

    Returns:
    Tuple[Tuple[float, float], Tuple[float, float]]: The bounds of the GeoTIFF file as a tuple of two tuples.
    The first tuple contains the (latitude, longitude) coordinates of the bottom left corner of the image, and the
    second tuple contains the (latitude, longitude) coordinates of the top right corner of the image.
    """
    dataset = rasterio.open(tif_path)
    b = dataset.bounds
    bounds = [(b.bottom, b.left), (b.top, b.right)]
    return bounds


def get_years_in_path(full_path: Path) -> List[str]:
    """
    Return a list of directory names within the given directory that match the pattern of a four-digit year (e.g. '2022').

    Args:
        directory: The directory to search for year folders. This can be a string or a Path object.

    Returns:
        A list of directory names within the given directory that match the pattern of a four-digit year.
    """
    full_path = Path(full_path)
    years = []
    with os.scandir(full_path) as entries:
        for entry in entries:
            if entry.is_dir() and re.match(r"^\d{4}$", entry.name):
                years.append(entry.name)
    return years


def find_file(dir_path, filename, case_insensitive=True):
    if case_insensitive:
        filename = filename.lower()
    for file in os.listdir(dir_path):
        if file.lower() == filename:
            return os.path.join(dir_path, file)
    return None


# @time_func
def get_image_overlay(
    tif_path,
    image_path,
    layer_name: str,
    convert_RGB: bool = True,
    file_format: str = "png",
):
    """
    Creates an image overlay for a GeoTIFF file using a JPG image.

    Parameters:
    tif_path (str): The file path of the GeoTIFF file.
    jpg_path (str): The file path of the JPG image.
    layer_name (str): The name of the layer.

    Returns:
    ImageOverlay: An image overlay for the GeoTIFF file.
    """
    logger.info(f"tif_path: {tif_path}")
    logger.info(f"image_path: {image_path}")
    logger.info(f"convert_RGB: {convert_RGB}")
    logger.info(f"file_format: {file_format}")
    bounds = get_bounds(tif_path)
    # convert image to RGB
    if convert_RGB:
        image_path = convert_to_rgb(image_path)

    image_overlay = map_functions.get_overlay_for_image(
        image_path, bounds, layer_name, file_format=file_format
    )
    return image_overlay


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        print(f"Elapsed time: {self.interval:.6f} seconds")


def build_tiff(tif_path, vrt_path):
    # then build tiff
    ds = gdal.Translate(
        destName=tif_path,
        creationOptions=["NUM_THREADS=ALL_CPUS", "COMPRESS=LZW", "TILED=YES"],
        srcDS=vrt_path,
    )
    ds.FlushCache()
    ds = None


def build_vrt(vrt_path: str, imgsToMosaic: List[str], resampleAlg: str = "mode"):
    vrt_options = gdal.BuildVRTOptions(resampleAlg=resampleAlg)
    try:
        ds = gdal.BuildVRT(vrt_path, imgsToMosaic, options=vrt_options)
        ds.FlushCache()
        ds = None
    except Exception as e:
        print(f"Error building VRT file: {e}")


def create_dir_chooser(callback, title: str = None, starting_directory: str = "data"):
    """
    Creates a file chooser widget for selecting directories and a button to close the file chooser.

    Parameters:
    callback (function): A function to be called when the user selects a directory.
    title (str): The title of the file chooser (optional).
    starting_directory (str): The starting directory of the file chooser (default is "data").

    Returns:
    HBox: A horizontal box containing the file chooser and close button.
    """
    padding = "0px 0px 0px 5px"  # upper, right, bottom, left
    inital_path = os.path.join(os.getcwd(), starting_directory)
    if not os.path.exists(inital_path):
        inital_path = os.getcwd()
    # creates a unique instance of filechooser and button to close filechooser
    dir_chooser = FileChooser(inital_path)
    dir_chooser.dir_icon = os.sep
    # Switch to folder-only mode
    dir_chooser.show_only_dirs = True
    if title is not None:
        dir_chooser.title = f"<b>{title}</b>"
    dir_chooser.register_callback(callback)

    close_button = ToggleButton(
        value=False,
        tooltip="Close Directory Chooser",
        icon="times",
        button_style="primary",
        layout=Layout(height="28px", width="28px", padding=padding),
    )

    def close_click(change):
        if change["new"]:
            dir_chooser.close()
            close_button.close()

    close_button.observe(close_click, "value")
    chooser = HBox([dir_chooser, close_button])
    return chooser


def group_tif_locations(dir_path):
    """
    Groups GeoTIFF files at the same location in a directory by their georeferencing information.

    Args:
        dir_path: str - The file path to the directory containing GeoTIFF files.

    Returns:
        list - A list of groups of file paths to GeoTIFF files that are at the same location.

    Example:
        groups = group_tif_locations("path/to/tif_directory")
        # Returns a list of groups of file paths to GeoTIFF files that are at the same location.
    """
    tif_files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith(".tif") or f.endswith(".tiff")
    ]

    tif_groups = {}

    for tif_file in tif_files:
        dataset = gdal.Open(tif_file, gdal.GA_ReadOnly)
        geo_transform = dataset.GetGeoTransform()
        x_min, x_size, _, y_max, _, y_size = geo_transform

        found_group = False
        for group_key in tif_groups.keys():
            if (x_min, x_size, y_max, y_size) == group_key:
                tif_groups[group_key].append(tif_file)
                found_group = True
                break

        if not found_group:
            tif_groups[(x_min, x_size, y_max, y_size)] = [tif_file]

    return list(tif_groups.values())


def group_files(files: List[str], size: int = 2) -> List[List[str]]:
    """
    Groups a list of file paths into sublists of a specified size.

    This function takes a list of file paths and groups them into sublists of a specified size. The default size is 2. The function returns a list of sublists, where each sublist contains up to `size` file paths.

    Parameters:
    - files (List[str]): A list of file paths to be grouped.
    - size (int): The size of each sublist. Defaults to 2.

    Returns:
    - A list of sublists, where each sublist contains up to `size` file paths.
    """
    grouped_files = [files[n : n + size] for n in range(0, len(files), size)]
    return grouped_files


def create_merged_multispectural_for_ROIs(roi_paths: List[str]) -> None:
    """
    Creates a merged multispectral image for each year of ROI data in the specified directories.

    Parameters:
    roi_paths (List[str]): A list of file paths to directories containing ROI data.

    Returns:
    None: This function does not return anything.
    """
    for roi_path in roi_paths:
        year_dirs = get_matching_dirs(roi_path, pattern=r"^\d{4}$")
        for year_path in year_dirs:
            glob_str = os.path.join(year_path, "*merged_multispectral.jpg")
            # A merged jpg has already been createed
            if len(glob(glob_str)) >= 1:
                logger.warning(f"*merged_multispectral.jpg already exists {year_path}")
                continue
            # no tifs exist to merge
            if len(os.listdir(year_path)) == 0:
                logger.warning(f"*{year_path} contains no tifs")
                continue
            try:
                # create merged_multispectral.jpg
                get_merged_multispectural(year_path)
            except Exception as merge_error:
                logger.error(f"Year: {year_path}\nmerge_error: {merge_error}")
                print(f"Year: {year_path}\nmerge_error: {merge_error}")
                continue


def get_merged_multispectural(src_path: str) -> str:
    """
    Merges multiple GeoTIFF files into a single VRT file and returns the path of the merged file.

    This function looks for all GeoTIFF files in the specified directory, except for any files that contain the string "merged_multispectral" in their name. It groups the remaining files into batches of 4, and merges each batch into a separate VRT file. Finally, it merges all the VRT files into a single VRT file and returns the path of the merged file.

    Parameters:
    - src_path (str): The path of the directory containing the GeoTIFF files.

    Returns:
    - The path of the merged VRT file.
    """
    # get all the unmerged tif files
    tif_files = glob(os.path.join(src_path, "*.tif"))
    tif_files = [file for file in tif_files if "merged_multispectral" not in file]

    logger.info(f"Found {len(tif_files)} GeoTIFF files in {src_path}")
    logger.info(f"tif_files: {tif_files}")

    vrt_path = os.path.join(src_path, "merged_multispectral.vrt")
    merged_file = merge_files(tif_files, vrt_path, create_jpg=True)
    return merged_file


def merge_files(src_files: str, vrt_path: str, create_jpg: bool = True) -> str:
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
        # create vrt(virtual world format) file
        vrt_options = gdal.BuildVRTOptions(resampleAlg="mode", srcNodata=0, VRTNodata=0)
        logger.info(f"dest_path: {vrt_path}")
        # creates a virtual world file using all the tifs and overwrites any pre-existing .vrt
        virtual_dataset = gdal.BuildVRT(vrt_path, src_files, options=vrt_options)
        # flushing the cache causes the vrt file to be created
        virtual_dataset.FlushCache()
        # reset the dataset object
        virtual_dataset = None

        # create geotiff (.tiff) from merged vrt file
        tif_path = vrt_path.replace(".vrt", ".tif")
        virtual_dataset = gdal.Translate(
            tif_path,
            creationOptions=["COMPRESS=LZW", "TILED=YES"],
            srcDS=vrt_path,
        )
        virtual_dataset.FlushCache()
        virtual_dataset = None

        if create_jpg:
            # convert .vrt to .jpg file
            virtual_dataset = gdal.Translate(
                vrt_path.replace(".vrt", ".jpg"),
                creationOptions=["WORLDFILE=YES", "QUALITY=100"],
                srcDS=tif_path,
            )
            virtual_dataset.FlushCache()
            virtual_dataset = None

        return vrt_path
    except Exception as e:
        print(e)
        logger.error(e)
        raise e


def delete_files(pattern, path,recursive=True):
    """
    Deletes all files in the directory tree rooted at `path` that match the given `pattern`.

    Args:
        pattern (str): Regular expression pattern to match against file names.
        path (str): Full path to the root of the directory tree to search for files.

    Returns:
        list: A list of full paths to the files that were deleted.

    Raises:
        ValueError: If the `path` does not exist.

    Example:
        # delete all the txt files in a directory tree
        >>> delete_files(r".+\.txt$", '/path/to/directory')
        ['/path/to/directory/file1.txt', '/path/to/directory/subdir/file2.txt']
    """
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")

    deleted_files = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if re.match(pattern, filename):
                    full_path = os.path.join(dirpath, filename)
                    os.remove(full_path)
                    deleted_files.append(full_path)
    else: # not recursive
        for filename in os.listdir(path):
            if re.match(pattern, filename):
                full_path = os.path.join(path, filename)
                os.remove(full_path)
                deleted_files.append(full_path)

    return deleted_files


def gdal_translate_png_to_tiff(
    files: List[str],
    translateoptions: str = "-of JPEG -co COMPRESS=JPEG -co TFW=YES -co QUALITY=100",
):
    """Convert TIFF files to JPEG files using GDAL.

    Args:
        files (List[str]): List of file paths to TIFF files to be converted.
        translateoptions (str, optional): GDAL options for converting TIFF files to JPEG files. Defaults to "-of JPEG -co COMPRESS=JPEG -co TFW=YES -co QUALITY=100".

    Returns:
        List[str]: List of file paths to the newly created JPEG files.
    """
    new_files = []
    for file in files:
        new_file = file.replace(".png", ".tif")
        if os.path.exists(new_file):
            logger.info(f"File: {new_file} already exists")
            print(f"File: {new_file} already exists")
        else:
            dst = gdal.Translate(new_file, file, options=translateoptions)
            new_files.append(new_file)
            dst = None  # close and save ds
    return new_files


def move_files_resurcively(src, dest):
    """Move all files and subdirectories from the source directory to the destination directory recursively.

    Args:
        source_dir (str): The path to the source directory.
        destination_dir (str): The path to the destination directory.

    Returns:
        None
    """
    if not os.path.isdir(src):
        os.makedirs(src)
    if not os.path.isdir(dest):
        os.makedirs(dest)
    # Move all files and subdirectories from the source directory to the destination directory
    for entry in os.scandir(src):
        # Move the entry to the destination directory
        shutil.move(entry.path, os.path.join(dest, entry.name))


def gdal_translate_jpegs(
    files: List[str],
    translateoptions: str = None,
    kwargs=None,
):
    """Convert TIFF files to JPEG files using GDAL.

    Args:
        files (List[str]): List of file paths to TIFF files to be converted.
        translateoptions (str, optional): GDAL options for converting TIFF files to JPEG files.
        kwargs(dict, optional): dictionary of GDAL options for converting TIFF files to JPEG files. Options located at:
            https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.TranslateOptions
    Returns:
        List[str]: List of file paths to the JPEG files.
    """
    jpg_files = []
    for file in files:
        jpg_file = file.replace(".tif", ".jpg")
        if os.path.exists(jpg_file):
            logger.info(f"File: {jpg_file} already exists")
            jpg_files.append(jpg_file)
        else:
            if kwargs:
                dst = gdal.Translate(jpg_file, file, **kwargs)
                jpg_files.append(jpg_file)
            elif translateoptions:
                dst = gdal.Translate(jpg_file, file, options=translateoptions)
                jpg_files.append(jpg_file)
            else:
                raise ValueError("Must provide value for kwargs or translateoptions.")
            dst = None  # close and save ds
    return jpg_files


def rename_files(directory: str, pattern: str, new_name: str, replace_name: str):
    """Rename all files in a directory that match a glob pattern

    Args:
        directory (str): the path to the directory containing the files to be renamed
        pattern (str): the glob pattern to match the files to be renamed
        new_name (str): the new prefix for the renamed files
    """
    # Get a list of files that match the pattern
    files = glob(os.path.join(directory, pattern))
    logger.info(f"Files to rename: {files}")

    for file in files:
        # Get the base name of the file
        base_name = os.path.basename(file)

        # Construct the new file name
        new_file_name = base_name.replace(replace_name, new_name)
        new_file_path = os.path.join(directory, new_file_name)

        # Rename the file
        os.rename(file, new_file_path)


def filter_files(files: List[str], avoid_patterns: List[str]) -> List[str]:
    """
    Filter a list of filepaths based on a list of avoid patterns.

    Args:
        files: A list of filepaths to filter.
        avoid_patterns: A list of regular expression patterns to avoid.

    Returns:
        A list of filepaths whose filenames do not match any of the patterns in avoid_patterns.

    Examples:
        >>> files = ['/path/to/file1.txt', '/path/to/file2.txt', '/path/to/avoid_file.txt']
        >>> avoid_patterns = ['.*avoid.*']
        >>> filtered_files = filter_files(files, avoid_patterns)
        >>> print(filtered_files)
        ['/path/to/file1.txt', '/path/to/file2.txt']

    """
    filtered_files = []
    for file in files:
        # Check if the file's name matches any of the avoid patterns
        for pattern in avoid_patterns:
            if re.match(pattern, os.path.basename(file)):
                break
        else:
            # If the file's name does not match any of the avoid patterns, add it to the filtered files list
            filtered_files.append(file)
    return filtered_files


def copy_files(
    src_files: List[str], dst_dir: str, avoid_patterns: List[str] = []
) -> None:
    """Copy files from a list of source files to a destination directory, while avoiding files with specific names.

    Args:
    src_files (List[str]): A list of file paths to be copied.
    dst_dir (str): The path to the destination directory.
    avoid_patterns (List[str], optional): A list of substrings to avoid in filenames. Defaults to [].

    Returns:
    None
    """
    logger.info(f"Copying files to {dst_dir}. Files: {src_files}")
    os.makedirs(dst_dir, exist_ok=True)
    files = filter_files(src_files, avoid_patterns)

    for src_file in files:
        dst_file = os.path.join(dst_dir, os.path.basename(src_file))
        shutil.copy(src_file, dst_file)


def move_files(src_dir: str, dst_dir: str, delete_src: bool = False) -> None:
    """
    Moves every file in a source directory to a destination directory, and has the option to delete the source directory when finished.

    The function uses the `shutil` library to move the files from the source directory to the destination directory. If the `delete_src` argument is set to `True`, the function will delete the source directory after all the files have been moved.

    Args:
    - src_dir (str): The path of the source directory.
    - dst_dir (str): The path of the destination directory.
    - delete_src (bool, optional): A flag indicating whether to delete the source directory after the files have been moved. Default is `False`.

    Returns:
    - None
    """
    logger.info(f"Moving files from {src_dir} to dst_dir. Delete Source:{delete_src}")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        dst_file = os.path.join(dst_dir, filename)
        shutil.move(src_file, dst_file)
    if delete_src:
        os.rmdir(src_dir)


def get_matching_dirs(dir_path: str, pattern: str = r"^\d{4}$") -> List[str]:
    """
    Returns a list of directories that match the specified pattern.

    The function searches the specified directory and its subdirectories for
    directories that have names that match the specified pattern.

    Args:
    - dir_path (str): The directory to search for matching directories.
    - pattern (str, optional): The pattern to match against the directory names. Default is "^\d{4}$".

    Returns:
    - List[str]: A list of the full paths of the matching directories.
    """
    matching_dirs = []
    for root, dirs, files in os.walk(dir_path):
        folder_name = os.path.basename(root)
        if re.match(pattern, folder_name):
            matching_dirs.append(root)
    return matching_dirs



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


def remove_zip(path) -> None:
    # Get a list of all the zipped files in the directory
    zipped_files = [
        os.path.join(path, f) for f in os.listdir(path) if f.endswith(".zip")
    ]
    # Remove each zip file
    for zipped_file in zipped_files:
        os.remove(zipped_file)


def unzip_dir(path: str):
    """
    Recursively unzips all the zip files in a given directory and its subdirectories.

    Args:
        path (str): The path to the directory to unzip.

    Returns:
        None

    Raises:
        zipfile.BadZipFile: If the zip file is corrupted or not a valid zip file.

    Note:
        This function assumes that the zip files are not password-protected.

    Example:
        >>> unzip_dir('/path/to/directory')

    """
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith(".zip"):
                file_path = os.path.join(root, filename)
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(root)
                os.remove(file_path)


def unzip(path) -> None:
    # Get a list of all the zipped files in the directory
    zipped_files = [
        os.path.join(path, f) for f in os.listdir(path) if f.endswith(".zip")
    ]
    logger.info(f"zipped_files:{zipped_files}")
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


def get_yearly_ranges(date_range):
    """
    Returns a list of start and end dates for each year in the specified date range.

    Parameters:
    - date_range (list): A list of two dates in the format ['YYYY-MM-DD', 'YYYY-MM-DD'].

    Returns:
    - A list of tuples, where each tuple contains the start and end date for a single year in the range.
    """
    start_date = datetime.strptime(date_range[0], "%Y-%m-%d")
    end_date = datetime.strptime(date_range[1], "%Y-%m-%d")
    year_ranges = []
    for year in range(start_date.year, end_date.year + 1):
        year_start = datetime(year, 1, 1)
        year_end = datetime(year, 12, 31)
        if year == start_date.year:
            year_start = start_date
        if year == end_date.year:
            year_end = end_date
        year_ranges.append((year_start, year_end))
    return year_ranges


def unzip_data(parent_dir: str):
    logger.info(f"Parent directory to find zip files: {parent_dir}")
    logger.info(f"All files in parent dir {os.listdir(parent_dir)}")
    subdirs = get_subdirs(parent_dir)
    logger.info(f"Subdirectories to unzip: {subdirs}")
    for subdir in subdirs:
        logger.info(f"SUBDIR {os.listdir(subdir)}")

    unzip_files(parent_dir)
    remove_zip_files(parent_dir)
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


def create_directory(file_path: str, name: str) -> str:
    """
    Creates a new directory with the given name in the specified file path, if it does not already exist.
    Returns the full path to the new directory.
    """
    new_directory = os.path.join(file_path, name)
    os.makedirs(new_directory, exist_ok=True)
    return new_directory


def generate_random_string(avoid_list=[]):
    alphanumeric = string.ascii_letters + string.digits
    random_string = "".join(random.choice(alphanumeric) for i in range(6))
    if random_string in avoid_list:
        return generate_random_string(avoid_list)
    return random_string


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


def delete_empty_dirs(dir_path: str):
    """
    Recursively delete all empty directories within a directory.

    Parameters
    ----------
    dir_path : str
        The path to the directory where the search for empty directories begins.

    Returns
    -------
    None
    """
    subdirs = get_subdirs(dir_path)
    remove_dirs = [subdir for subdir in subdirs if len(os.listdir(subdir)) == 0]
    for remove_dir in remove_dirs:
        os.removedirs(remove_dir)


def create_year_directories(start_year: int, end_year: int, base_path: str) -> None:
    """Create directories for each year in between a given start and end year.

    Args:
        start_year (int): The start year.
        end_year (int): The end year.
        base_path (str): The base path for the directories.

    Returns:
        None
    """
    for year in range(start_year, end_year + 1):
        year_path = os.path.join(base_path, str(year))
        if not os.path.exists(year_path):
            os.makedirs(year_path)


def create_subdirectory(name: str, parent_dir: str = None) -> str:
    """Returns full path to a directory named name created in the parent directory.
    If the parent directory is not given then the data directory is created in the current working directory

    Args:
        parent_dir (str, optional): parent directory to create name directory within. Defaults to None.

    Returns:
        str: full path to a directory named name
    """
    if parent_dir == None:
        parent_dir = os.getcwd()
    new_dir = os.path.join(parent_dir, name)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    return new_dir


def create_warning_box(title: str = None, msg: str = None):
    padding = "0px 5px 0px 5px"  # upper, right, bottom, left
    # create title
    if title is None:
        title = "Warning"
    warning_title = HTML(f"<b>⚠️<u>{title}</u></b>")
    # create msg
    if msg is None:
        msg = "Something went wrong..."
    warning_msg = HTML(
        f"____________________________________________________________________________________________\
                   </br>⚠️{msg}"
    )
    # create vertical box to hold title and msg
    warning_content = VBox([warning_title, warning_msg])
    # define a close button
    close_button = ToggleButton(
        value=False,
        tooltip="Close Warning Box",
        icon="times",
        button_style="danger",
        layout=Layout(height="28px", width="28px", padding=padding),
    )
    ok_button = ToggleButton(
        value=False,
        tooltip="Close Warning Box",
        description="OK",
        button_style="danger",
        layout=Layout(height="28px", width="28px", padding="0px 0px 0px 0px"),
    )

    def close_click(change):
        if change["new"]:
            warning_content.close()
            close_button.close()
            ok_button.close()

    ok_button.observe(close_click, "value")
    close_button.observe(close_click, "value")
    warning_box = HBox([VBox([warning_content, ok_button]), close_button])
    return warning_box


def clear_row(row: HBox):
    """close widgets in row/column and clear all children
    Args:
        row (HBox)(VBox): row or column
    """
    for index in range(len(row.children)):
        row.children[index].close()
    row.children = []


def save_to_geojson_file(out_file: str, geojson: dict, **kwargs) -> None:
    """save_to_geojson_file Saves given geojson to a geojson file at outfile
    Args:
        out_file (str): The output file path
        geojson (dict): geojson dict containing FeatureCollection for all geojson objects in selected_set
    """
    # Save the geojson to a file
    out_file = check_file_path(out_file)
    ext = os.path.splitext(out_file)[1].lower()
    if ext == ".geojson":
        out_geojson = out_file
    else:
        out_geojson = os.path.splitext(out_file)[1] + ".geojson"
    with open(out_geojson, "w") as f:
        json.dump(geojson, f, **kwargs)


def download_url(url: str, save_path: str, filename: str = None, chunk_size: int = 128):
    """Downloads the data from the given url to the save_path location.
    Args:
        url (str): url to data to download
        save_path (str): directory to save data
        chunk_size (int, optional):  Defaults to 128.
    """
    with requests.get(url, stream=True) as r:
        if r.status_code == 404:
            logger.error(f"DownloadError: {save_path}")
            raise exceptions.DownloadError(os.path.basename(save_path))
        # check header to get content length, in bytes
        total_length = int(r.headers.get("Content-Length"))
        with open(save_path, "wb") as fd:
            with tqdm(
                total=total_length,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {filename}",
                initial=0,
                ascii=True,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    fd.write(chunk)
                    pbar.update(len(chunk))


def is_list_empty(main_list: list) -> bool:
    all_empty = True
    for np_array in main_list:
        if len(np_array) != 0:
            all_empty = False
    return all_empty


def get_center_rectangle(coords: list) -> tuple:
    """returns the center point of rectangle specified by points coords
    Args:
        coords list[tuple(float,float)]: lat,lon coordinates
    Returns:
        tuple[float]: (center x coordinate, center y coordinate)
    """
    x1, y1 = coords[0][0], coords[0][1]
    x2, y2 = coords[2][0], coords[2][1]
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    return center_x, center_y


def get_epsg_from_geometry(geometry: "shapely.geometry.polygon.Polygon") -> int:
    """Uses geometry of shapely rectangle in crs 4326 to return the most accurate
    utm code as a string of format 'epsg:utm_code'
    example: 'espg:32610'

    Args:
        geometry (shapely.geometry.polygon.Polygon): geometry of a rectangle

    Returns:
        int: most accurate epsg code based on lat lon coordinates of given geometry
    """
    rect_coords = geometry.exterior.coords
    center_x, center_y = get_center_rectangle(rect_coords)
    utm_code = convert_wgs_to_utm(center_x, center_y)
    return int(utm_code)


def convert_wgs_to_utm(lon: float, lat: float) -> str:
    """return most accurate utm epsg-code based on lat and lng
    convert_wgs_to_utm function, see https://stackoverflow.com/a/40140326/4556479
    Args:
        lon (float): longitude
        lat (float): latitude
    Returns:
        str: new espg code
    """
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == "1":
        utm_band = "0" + utm_band
    if lat >= 0:
        epsg_code = "326" + utm_band  # North
        return epsg_code
    epsg_code = "327" + utm_band  # South
    return epsg_code


def get_colors(length: int) -> list:
    # returns a list of color hex codes as long as length
    cmap = matplotlib.pyplot.get_cmap("plasma", length)
    cmap_list = [matplotlib.colors.rgb2hex(i) for i in cmap.colors]
    return cmap_list


def get_area(polygon: dict) -> float:
    "Calculates the area of the geojson polygon using the same method as geojson.io"
    logger.info(f"get_area: {polygon}")
    return round(area(polygon), 3)


def read_json_file(filename: str) -> dict:
    with open(filename, "r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    return data


def get_ids_with_invalid_area(
    geometry: gpd.GeoDataFrame, max_area: float = 98000000, min_area: float = 0
) -> set:
    if isinstance(geometry, gpd.GeoDataFrame):
        geometry = json.loads(geometry.to_json())
    if isinstance(geometry, dict):
        if "features" in geometry.keys():
            rows_drop = set()
            for i, feature in enumerate(geometry["features"]):
                roi_area = get_area(feature["geometry"])
                if roi_area >= max_area or roi_area <= min_area:
                    rows_drop.add(i)
            return rows_drop
    else:
        raise TypeError("Must be geodataframe")


def find_config_json(search_path: str, search_pattern: str = None) -> str:
    """Searches for a `config.json` file in the specified directory

    Args:
        search_path (str): the directory path to search for the `config.json` file

    Returns:
        str: the file path to the `config.json` file

    Raises:
        FileNotFoundError: if a `config.json` file is not found in the specified directory
    """
    logger.info(f"searching directory for config.json: {search_path}")
    if search_pattern == None:
        search_pattern = r"^config\.json$"
    config_regex = re.compile(search_pattern, re.IGNORECASE)
    logger.info(f"search_pattern: {search_pattern}")

    for file in os.listdir(search_path):
        if config_regex.match(file):
            logger.info(f"{file} matched regex")
            file_path = os.path.join(search_path, file)
            return file_path

    raise FileNotFoundError(f"config.json file was not found at {search_path}")


def config_to_file(config: Union[dict, gpd.GeoDataFrame], file_path: str):
    """Saves config to config.json or config_gdf.geojson
    config's type is dict or geodataframe respectively

    Args:
        config (Union[dict, gpd.GeoDataFrame]): data to save to config file
        file_path (str): full path to directory to save config file
    """
    if isinstance(config, dict):
        filename = f"config.json"
        save_path = os.path.abspath(os.path.join(file_path, filename))
        write_to_json(save_path, config)
        logger.info(f"Saved config json: {filename} \nSaved to {save_path}")
    elif isinstance(config, gpd.GeoDataFrame):
        filename = f"config_gdf.geojson"
        save_path = os.path.abspath(os.path.join(file_path, filename))
        logger.info(f"Saving config gdf:{config} \nSaved to {save_path}")
        config.to_file(save_path, driver="GeoJSON")


def create_json_config(input_settings: dict, settings: dict) -> dict:
    """returns config dictionary with the settings, currently selected_roi ids, and
    each of the input_settings specified by roi id.
    sample config:
    {
        'roi_ids': ['17','20']
        'settings':{ 'dates': ['2018-12-01', '2019-03-01'],
                    'sitename':'sitename1'}
        '17':{
            'sat_list': ['L8'],
            'landsat_collection': 'C01',
            'dates': ['2018-12-01', '2019-03-01'],
            'sitename':'roi_17',
            'filepath':'C:\\Home'
        }
        '20':{
            'sat_list': ['L8'],
            'landsat_collection': 'C01',
            'dates': ['2018-12-01', '2019-03-01'],
            'sitename':'roi_20',
            'filepath':'C:\\Home'
        }
    }

    Args:
        input_settings (dict): json style dictionary with roi ids at the keys with input_settings as values
        settings (dict):  json style dictionary containing map settings
    Returns:
        dict: json style dictionary, config
    """
    roi_ids = list(input_settings.keys())
    config = {**input_settings}
    config["roi_ids"] = roi_ids
    config["settings"] = settings
    return config


def create_config_gdf(
    rois: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Returns a new geodataframe with new column "type" that contains feature name.
        feature name is one of the following: "roi"

    Args:
        rois (gpd.GeoDataFrame,): geodataframe of rois

    Returns:
        gpd.GeoDataFrame: new geodataframe with new column "type" that contains feature name.
        feature name is one of the following: "roi"
    """
    # create new column 'type' to indicate object type
    rois["type"] = "roi"
    new_gdf = gpd.GeoDataFrame(rois)
    return new_gdf


def write_to_json(filepath: str, settings: dict):
    """ "Write the  settings dictionary to json file"""
    with open(filepath, "w", encoding="utf-8") as output_file:
        json.dump(settings, output_file)


def read_geojson_file(geojson_file: str) -> dict:
    """Returns the geojson of the selected ROIs from the file specified by geojson_file"""
    with open(geojson_file) as f:
        data = geojson.load(f)
    return data


def read_gpd_file(filename: str) -> gpd.GeoDataFrame:
    """
    Returns geodataframe from geopandas geodataframe file
    """
    if os.path.exists(filename):
        logger.info(f"Opening \n {filename}")
        with open(filename, "r") as f:
            gpd_data = gpd.read_file(f)
    else:
        raise FileNotFoundError
    return gpd_data


def create_roi_settings(
    settings: dict,
    selected_ids: set,
    filepath: str,
) -> dict:
    """returns a dict of settings for each roi with roi id as the key.
    Example:
    "2": {
            "dates": ["2018-12-01", "2019-03-01"],
            "sitename": "roi",
            "roi_name": "ID_2_dates_2010-01-01_to_2010-12-31",
            "filepath": "C:\\CoastSeg\\data",
            "roi_id": "2",
        },
    "3": {
            "dates": ["2018-12-01", "2019-03-01"],
            "sitename": "sitename1",
            "roi_name": "ID_3_dates_2010-01-01_to_2010-12-31"
            "filepath": "C:\\CoastSeg\\data",
            "roi_id": "3",
        },

    Args:
        settings (dict): currently loaded settings for the map
        selected_ids (set): set of selected ROI ids
        filepath (str): full path to data directory
    Returns:
        dict: settings for each roi with roi id as the key
    """

    roi_settings = {}
    sitename = settings["sitename"]
    dates = settings["dates"]
    for roi_id in list(selected_ids):
        roi_name = f"ID_{roi_id}_dates_{dates[0]}_to_{dates[1]}"
        roi_info = {
            "dates": dates,
            "roi_id": roi_id,
            "roi_name": roi_name,
            "sitename": sitename,
            "filepath": filepath,
        }
        roi_settings[roi_id] = roi_info
    return roi_settings


def do_rois_filepaths_exist(roi_settings: dict, roi_ids: list) -> bool:
    """Returns true if all rois have filepaths that exist
    Args:
        roi_settings (dict): settings of all rois on map
        roi_ids (list): ids of rois selected on map
    Returns:
        bool: True if all rois have filepaths that exist
    """
    # by default assume all filepaths exist
    does_filepath_exist = True
    for roi_id in roi_ids:
        filepath = str(roi_settings[roi_id]["filepath"])
        if not os.path.exists(filepath):
            # if filepath does not exist stop checking
            does_filepath_exist = False
            logger.info(f"filepath did not exist{filepath}")
            print("Some ROIs contained filepaths that did not exist")
            break
    logger.info(f"{does_filepath_exist} All rois filepaths exist")
    return does_filepath_exist


def do_rois_dirs_exist(roi_settings: dict, roi_ids: list) -> bool:
    """Returns true if all rois have directories that exist
    Args:
        roi_settings (dict): settings of all rois on map
        roi_ids (list): ids of rois selected on map
    Returns:
        bool: True if all rois have filepaths that exist
    """
    # by default assume all filepaths exist
    does_filepath_exist = True
    for roi_id in roi_ids:
        if "filepath" not in roi_settings[roi_id].keys():
            does_filepath_exist = False
            logger.info(f"roi_path did not exist because no filepath found")
            print("Some ROIs contained directories that did not exist")
            break
        if "sitename" not in roi_settings[roi_id].keys():
            does_filepath_exist = False
            logger.info(f"roi_path did not exist because no sitename found")
            print("Some ROIs contained directories that did not exist")
            break
        filepath = os.path.abspath(roi_settings[roi_id]["filepath"])
        sitename = roi_settings[roi_id]["sitename"]
        roi_name = roi_settings[roi_id]["roi_name"]
        roi_path = os.path.join(filepath, sitename, roi_name)
        if not os.path.exists(roi_path):
            # if filepath does not exist stop checking
            does_filepath_exist = False
            logger.info(f"roi_path did not exist{roi_path}")
            print("Some ROIs contained directories that did not exist")
            break
    logger.info(f"{does_filepath_exist} All rois directories exist")
    return does_filepath_exist


def do_rois_have_sitenames(roi_settings: dict, roi_ids: list) -> bool:
    """Returns true if all rois have "sitename" with non-empty string
    Args:
        roi_settings (dict): settings of all rois on map
        roi_ids (list): ids of rois selected on map
    Returns:
        bool: True if all rois have "sitename" with non-empty string
    """
    # by default assume all sitenames are not empty
    is_sitename_not_empty = True
    for roi_id in roi_ids:
        if roi_settings[roi_id]["sitename"] == "":
            # if sitename is empty means user has not downloaded ROI data
            is_sitename_not_empty = False
            break
    logger.info(f"{is_sitename_not_empty} All rois have non-empty sitenames")
    return is_sitename_not_empty


def were_rois_downloaded(roi_settings: dict, roi_ids: list) -> bool:
    """Returns true if rois were downloaded before. False if they have not
    Uses 'sitename' key for each roi to determine if roi was downloaded.
    And checks if filepath were roi is saved is valid
    If each roi's 'sitename' is not empty string returns true
    Args:
        roi_settings (dict): settings of all rois on map
        roi_ids (list): ids of rois selected on map
    Returns:
        bool: True means rois were downloaded before
    """
    # by default assume rois were downloaded
    is_downloaded = True
    if roi_settings is None:
        # if rois do not have roi_settings this means they were never downloaded
        is_downloaded = False
    elif roi_settings == {}:
        # if rois do not have roi_settings this means they were never downloaded
        is_downloaded = False
    elif roi_settings != {}:
        all_sitenames_exist = do_rois_have_sitenames(roi_settings, roi_ids)
        all_filepaths_exist = do_rois_filepaths_exist(roi_settings, roi_ids)
        all_roi_dirs_exist = do_rois_dirs_exist(roi_settings, roi_ids)
        logger.info(
            f"all_filepaths_exist: {all_filepaths_exist} all_sitenames_exist{all_sitenames_exist}"
        )
        is_downloaded = (
            all_sitenames_exist and all_filepaths_exist and all_roi_dirs_exist
        )
    # print correct message depending on whether ROIs were downloaded
    if is_downloaded:
        logger.info(f"Located previously downloaded ROI data.")
    elif is_downloaded == False:
        print(
            "Did not locate previously downloaded ROI data. To download the imagery for your ROIs click Download Imagery"
        )
        logger.info(
            f"Did not locate previously downloaded ROI data. To download the imagery for your ROIs click Download Imagery"
        )
    return is_downloaded


def get_site_path(settings: dict) -> str:
    """
    Create a subdirectory with the name `settings["sitename"]` within a "data" directory in the current working
    directory to hold all downloads. If the subdirectory already exists, raise an error.

    Args:
    - settings: A dictionary containing the key `"sitename"`, which specifies the name of the subdirectory to be created.

    Returns:
    - The absolute file path of the newly created subdirectory.
    """
    # create data directory in current working directory to hold all downloads if it doesn't already exist
    data_path = create_subdirectory("data")
    # create sitename directory if it doesn't already exist
    site_path = os.path.join(data_path, settings["sitename"])
    # exception_handler.check_path_already_exists(site_path, settings["sitename"])
    if not os.path.exists(site_path):
        os.makedirs(site_path)
    return site_path


def generate_datestring() -> str:
    """Returns a datetime string in the following format %m-%d-%y__%I_%M_%S
    EX: "ID_0__01-31-22_12_19_45"""
    date = datetime.now()
    return date.strftime("%m-%d-%y__%I_%M_%S")


def mk_new_dir(name: str, location: str):
    """Create new folder with name_datetime stamp at location
    Args:
        name (str): name of folder to create
        location (str): full path to location to create folder
    """
    if os.path.exists(location):
        new_folder = location + os.sep + name + "_" + generate_datestring()
        os.mkdir(new_folder)
        return new_folder
    else:
        raise Exception("Location provided does not exist.")


def get_RGB_in_path(current_path: str) -> str:
    """returns full path to RGB directory relative to current path
    or raises an error

    Args:
        current_path (str): full path to directory of images to segment

    Raises:
        Exception: raised if no RGB directory is found or
        RGB directory is empty

    Returns:
        str: full path to RGB directory relative to current path
    """
    rgb_jpgs = glob(current_path + os.sep + "*RGB*")
    logger.info(f"rgb_jpgs: {rgb_jpgs}")
    if rgb_jpgs != []:
        return current_path
    elif rgb_jpgs == []:
        # means current path is not RGB directory
        parent_dir = os.path.dirname(current_path)
        logger.info(f"parent_dir: {parent_dir}")
        dirs = os.listdir(parent_dir)
        logger.info(f"child dirs: {dirs}")
        if "RGB" not in dirs:
            raise Exception(
                "Invalid directory to run model in. Please select RGB directory"
            )
        RGB_path = os.path.join(parent_dir, "RGB")
        logger.info(f"returning path:{RGB_path}")
        return RGB_path


def scale(matrix: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """returns resized matrix with shape(rows,cols)
        for 2d discrete labels
        for resizing 2d integer arrays
    Args:
        im (np.ndarray): 2d matrix to resize
        nR (int): number of rows to resize 2d matrix to
        nC (int): number of columns to resize 2d matrix to

    Returns:
        np.ndarray: resized matrix with shape(rows,cols)
    """
    src_rows = len(matrix)  # source number of rows
    src_cols = len(matrix[0])  # source number of columns
    tmp = [
        [
            matrix[int(src_rows * r / rows)][int(src_cols * c / cols)]
            for c in range(cols)
        ]
        for r in range(rows)
    ]
    return np.array(tmp).reshape((rows, cols))


def rescale_array(dat, mn, mx):
    """
    rescales an input dat between mn and mx
    Code from doodleverse_utils by Daniel Buscombe
    source: https://github.com/Doodleverse/doodleverse_utils
    """
    m = min(dat.flatten())
    M = max(dat.flatten())
    return (mx - mn) * (dat - m) / (M - m) + mn
