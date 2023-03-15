import os
import re
import shutil
import asyncio
import platform
import json
import logging
from typing import List, Set

from seg2map import common

import requests
import skimage
import aiohttp
import tqdm
import numpy as np
from glob import glob
from osgeo import gdal
import tqdm.asyncio
import nest_asyncio
from skimage.io import imread
from tensorflow.keras import mixed_precision
from doodleverse_utils.prediction_imports import do_seg
from doodleverse_utils.model_imports import (
    simple_resunet,
    custom_resunet,
    custom_unet,
    simple_unet,
    simple_resunet,
    simple_satunet,
    segformer,
)
from doodleverse_utils.model_imports import dice_coef_loss, iou_multi, dice_multi
import tensorflow as tf

logger = logging.getLogger(__name__)



def get_sorted_files_with_extension(
    sample_direc: str, file_extensions: List[str]
) -> List[str]:
    """
    Get a sorted list of paths to files that have one of the file_extensions.
    It will return the first set of files that matches the first file_extension, so put the
    file_extension list in order of priority

    Args:
        sample_direc: A string representing the directory path to search for images.
        file_extensions: A list of file extensions to search for.

    Returns:
        A list of file paths for sample images found in the directory.

    """
    sample_filenames = []
    for ext in file_extensions:
        filenames = sorted(tf.io.gfile.glob(os.path.join(sample_direc, f"*{ext}")))
        sample_filenames.extend(filenames)
        if sample_filenames:
            break
    return sample_filenames


def get_merged_multispectural(src_path: str) -> str:
    """
    Merges multiple GeoTIFF files into a single VRT file and returns the path of the merged file.

    This function looks for all GeoTIFF files in the specified directory, except for any files that contain the string "merged_multispectral" in their name. It groups the remaining files into batches of 4, and merges each batch into a separate VRT file. Finally, it merges all the VRT files into a single VRT file and returns the path of the merged file.

    Parameters:
    - src_path (str): The path of the directory containing the GeoTIFF files.

    Returns:
    - The path of the merged VRT file.
    """
    tif_files = glob(os.path.join(src_path, "*.tif"))
    tif_files = common.filter_files(tif_files, [r".*merged_multispectral.*"])
    logger.info(
        f"Found {len(tif_files)} GeoTIFF files in {src_path}.\n tif_files: {tif_files}"
    )
    dst_path = os.path.join(src_path, "merged_multispectral.vrt")
    merged_file = common.merge_files(tif_files, dst_path)

    return merged_file


def get_seg_files_by_year(dir_path: str) -> dict:
    """
    Returns a dictionary of segmentation files organized by year.

    The function searches the specified directory and its subdirectories for
    directories that are named as 4-digit years. For each year directory, it
    finds all `.jpg` files that do not contain the string "merged_multispectral".

    The returned dictionary has the year as the key and a dictionary as the value.
    The value dictionary has two keys: "file_path" and "jpgs". "file_path" is the
    full path of the year directory, and "jpgs" is a list of the full paths of
    the `.jpg` files that meet the criteria.

    Args:
    - dir_path (str): The directory to search for segmentation files.

    Returns:
    - dict: A dictionary of segmentation files organized by year.
    """
    files_per_year = {}
    for root, dirs, files in os.walk(dir_path):
        if root == dir_path:
            continue
        folder_name = os.path.basename(root)
        if not re.match(r"^\d{4}$", folder_name):
            continue
        jpg_paths = sorted(tf.io.gfile.glob(os.path.join(root, "*.jpg")))
        jpg_paths = [file for file in jpg_paths if "merged_multispectral" not in file]
        files_per_year[folder_name] = {"file_path": root, "jpgs": jpg_paths}
    return files_per_year


def get_five_band_imagery(RGB_path: str, MNDWI_path: str, NDWI_path: str, output_path: str) -> str:
    """Create a five-band image by combining three image directories containing JPEG files.

    Args:
        RGB_path (str): Path to directory containing red-green-blue (RGB) image files.
        MNDWI_path (str): Path to directory containing Modified Normalized Difference Water Index (MNDWI) image files.
        NDWI_path (str): Path to directory containing Normalized Difference Water Index (NDWI) image files.
        output_path (str): Path to directory where the output five-band image file will be saved.

    Returns:
        str: The path to the output directory where the compressed npz file has been saved.

    Raises:
        ValueError: If the file format is not recognized.

    """
    paths = [RGB_path, MNDWI_path, NDWI_path]
    files = []
    for data_path in paths:
        f = sorted(glob(data_path + os.sep + "*.jpg"))
        if len(f) < 1:
            f = sorted(glob(data_path + os.sep + "images" + os.sep + "*.jpg"))
        files.append(f)

    # number of bands x number of samples
    files = np.vstack(files).T
    # returns path to five band imagery
    for counter, file in enumerate(files):
        im = []  # read all images into a list
        for k in file:
            im.append(imread(k))
        datadict = {}
        # create stack which takes care of different sized inputs
        im = np.dstack(im)
        datadict["arr_0"] = im.astype(np.uint8)
        datadict["num_bands"] = im.shape[-1]
        datadict["files"] = [file_name.split(os.sep)[-1] for file_name in file]
        ROOT_STRING = file[0].split(os.sep)[-1].split(".")[0]
        segfile = (
            output_path
            + os.sep
            + ROOT_STRING
            + "_noaug_nd_data_000000"
            + str(counter)
            + ".npz"
        )
        np.savez_compressed(segfile, **datadict)
        del datadict, im
        logger.info(f"segfile: {segfile}")
    return output_path


def get_files(RGB_dir_path: str, img_dir_path: str):
    """returns matrix of files in RGB_dir_path and img_dir_path
    creates matrix: RGB x number of samples in img_dir_path
    Example:
    [['full_RGB_path.jpg','full_NIR_path.jpg'],
    ['full_jpg_path.jpg','full_NIR_path.jpg']....]
    Args:
        RGB_dir_path (str): full path to directory of RGB images
        img_dir_path (str): full path to directory of non-RGB images
        usually NIR and SWIR

    Raises:
        FileNotFoundError: raised if directory is not found
    """
    paths = [RGB_dir_path, img_dir_path]
    files = []
    for data_path in paths:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} not found")
        f = sorted(glob(data_path + os.sep + "*.jpg"))
        if len(f) < 1:
            f = sorted(glob(data_path + os.sep + "images" + os.sep + "*.jpg"))
        files.append(f)
    # creates matrix:  bands(RGB) x number of samples
    files = np.vstack(files).T
    return files


def RGB_to_infrared(
    RGB_path: str, infrared_path: str, output_path: str, output_type: str
) -> None:
    """Converts two directories of RGB and NIR imagery to NDWI imagery in a directory named
     'NDWI' created at output_path.
     imagery saved as jpg

     to generate NDWI imagery set infrared_path to full path of NIR images
     to generate MNDWI imagery set infrared_path to full path of SWIR images

    Args:
        RGB_path (str): full path to directory containing RGB images
        infrared_path (str): full path to directory containing NIR or SWIR images
        output_path (str): full path to directory to create NDWI/MNDWI directory in
        output_type (str): 'MNDWI' or 'NDWI'
    Based on code from doodleverse_utils by Daniel Buscombe
    source: https://github.com/Doodleverse/doodleverse_utils
    """
    if output_type.upper() not in ["MNDWI", "NDWI"]:
        logger.error(
            f"Invalid output_type given must be MNDWI or NDWI. Cannot be {output_type}"
        )
        raise Exception(
            f"Invalid output_type given must be MNDWI or NDWI. Cannot be {output_type}"
        )
    # matrix:bands(RGB) x number of samples(NIR)
    files = get_files(RGB_path, infrared_path)
    # output_path: directory to store MNDWI or NDWI outputs
    output_path += os.sep + output_type.upper()
    logger.info(f"output_path {output_path}")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for file in files:
        # Read green band from RGB image and cast to float
        green_band = skimage.io.imread(file[0])[:, :, 1].astype("float")
        # Read infrared(SWIR or NIR) and cast to float
        infrared = skimage.io.imread(file[1]).astype("float")
        # Transform 0 to np.NAN
        green_band[green_band == 0] = np.nan
        infrared[infrared == 0] = np.nan
        # Mask out NaNs
        green_band = np.ma.filled(green_band)
        infrared = np.ma.filled(infrared)

        # ensure both matrices have equivalent size
        if not np.shape(green_band) == np.shape(infrared):
            gx, gy = np.shape(green_band)
            nx, ny = np.shape(infrared)
            # resize both matrices to have equivalent size
            green_band = common.scale(
                green_band, np.maximum(gx, nx), np.maximum(gy, ny)
            )
            infrared = common.scale(infrared, np.maximum(gx, nx), np.maximum(gy, ny))

        # output_img(MNDWI/NDWI) imagery formula (Green - SWIR) / (Green + SWIR)
        output_img = np.divide(infrared - green_band, infrared + green_band)
        # Convert the NaNs to -1
        output_img[np.isnan(output_img)] = -1
        # Rescale to be between 0 - 255
        output_img = common.rescale_array(output_img, 0, 255)
        # create new filenames by replacing image type(SWIR/NIR) with output_type
        if output_type.upper() == "MNDWI":
            new_filename = file[1].split(os.sep)[-1].replace("SWIR", output_type)
        if output_type.upper() == "NDWI":
            new_filename = file[1].split(os.sep)[-1].replace("NIR", output_type)

        # save output_img(MNDWI/NDWI) as .jpg in output directory
        skimage.io.imsave(
            output_path + os.sep + new_filename,
            output_img.astype("uint8"),
            check_contrast=False,
            quality=100,
        )

    return output_path


async def async_download_url(session, url: str, save_path: str):
    model_name = url.split("/")[-1]
    # chunk_size: int = 128
    chunk_size: int = 2048
    async with session.get(url, raise_for_status=True) as r:
        content_length = r.headers.get("Content-Length")
        if content_length is not None:
            content_length = int(content_length)
            with open(save_path, "wb") as fd:
                with tqdm.auto.tqdm(
                    total=content_length,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {model_name}",
                    initial=0,
                    ascii=False,
                    position=0,
                ) as pbar:
                    async for chunk in r.content.iter_chunked(chunk_size):
                        fd.write(chunk)
                        pbar.update(len(chunk))
        else:
            with open(save_path, "wb") as fd:
                async for chunk in r.content.iter_chunked(chunk_size):
                    fd.write(chunk)


async def async_download_urls(url_dict):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for save_path, url in url_dict.items():
            task = asyncio.create_task(async_download_url(session, url, save_path))
            tasks.append(task)
        await tqdm.asyncio.tqdm.gather(*tasks)


def run_async_download(url_dict: dict):
    logger.info("run_async_download")
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # apply a nested loop to jupyter's event loop for async downloading
    nest_asyncio.apply()
    # get nested running loop and wait for async downloads to complete
    loop = asyncio.get_running_loop()
    result = loop.run_until_complete(async_download_urls(url_dict))
    logger.info(f"result: {result}")


def get_GPU(num_GPU: str) -> None:
    num_GPU = str(num_GPU)
    if num_GPU == "0":
        logger.info("Not using GPU")
        print("Not using GPU")
        # use CPU (not recommended):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif num_GPU == "1":
        print("Using single GPU")
        logger.info(f"Using 1 GPU")
        # use first available GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if int(num_GPU) == 1:
        # read physical GPUs from machine
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        print(f"physical_devices (GPUs):{physical_devices}")
        logger.info(f"physical_devices (GPUs):{physical_devices}")
        if physical_devices:
            # Restrict TensorFlow to only use the first GPU
            try:
                tf.config.experimental.set_visible_devices(physical_devices, "GPU")
            except RuntimeError as e:
                # Visible devices must be set at program startup
                logger.error(e)
                print(e)
        # disable memory growth on all GPUs
        for i in physical_devices:
            tf.config.experimental.set_memory_growth(i, True)
            print(f"visible_devices: {tf.config.get_visible_devices()}")
            logger.info(f"visible_devices: {tf.config.get_visible_devices()}")
        # if multiple GPUs are used use mirror strategy
        if int(num_GPU) > 1:
            # Create a MirroredStrategy.
            strategy = tf.distribute.MirroredStrategy(
                [p.name.split("/physical_device:")[-1] for p in physical_devices],
                cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(),
            )
            print(f"Number of distributed devices: {strategy.num_replicas_in_sync}")
            logger.info(
                f"Number of distributed devices: {strategy.num_replicas_in_sync}"
            )


def get_url_dict_to_download(models_json_dict: dict) -> dict:
    """Returns dictionary of paths to save files to download
    and urls to download file. If any of the files already exist then
    they will not be return in the dictionary.

    ex.
    {'C:\Home\Project\file.json':'https://website/file.json'}

    Args:
        models_json_dict (dict): full path to files and links

    Returns:
        dict: full path to files and links
    """
    # empty dictionary of files to be downloaded
    url_dict = {}
    # iterate through each full path to each file
    for save_path, link in models_json_dict.items():
        # if file doesn't exist add to dictionary of links to download
        if not os.path.isfile(save_path):
            logger.info(f"Did not exist save_path: {save_path}")
            url_dict[save_path] = link
        # check if json file exists for the same model (.h5) and if not add to dictionary of links to download
        if re.search(r"_fullmodel\.h5$", save_path):
            json_filepath = save_path.replace("_fullmodel.h5", ".json")
            if not os.path.isfile(json_filepath):
                json_link = link.replace("_fullmodel.h5", ".json")
                url_dict[json_filepath] = json_link
    return url_dict


def download_url(url: str, save_path: str, chunk_size: int = 128):
    """Downloads the model from the given url to the save_path location.
    Args:
        url (str): url to model to download
        save_path (str): directory to save model
        chunk_size (int, optional):  Defaults to 128.
    """
    logger.info(f"url: {url}")
    logger.info(f"save_path: {save_path}")
    # make an HTTP request within a context manager
    with requests.get(url, stream=True) as r:
        # check header to get content length, in bytes
        content_length = r.headers.get("Content-Length")
        # raise an exception for error codes (4xx or 5xx)
        r.raise_for_status()
        if content_length is None:
            with open(save_path, "wb") as fd:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    fd.write(chunk)
        elif content_length is not None:
            content_length = int(content_length)
            with open(save_path, "wb") as fd:
                with tqdm.auto.tqdm(
                    total=content_length,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Downloading Model",
                    initial=0,
                    ascii=True,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        fd.write(chunk)
                        pbar.update(len(chunk))


class Zoo_Model:
    def __init__(self):
        self.weights_direc = None

    def get_files_for_seg(
        self, sample_direc: str, avoid_patterns: List[str] = []
    ) -> list:
        """
        Returns a list of files to be segmented.

        The function reads in the image filenames as either (`.npz`) OR (`.jpg`, or `.png`)
        and returns a sorted list of the file paths.

        Args:
        - sample_direc (str): The directory containing files to be segmented.
        - avoid_patterns (List[str], optional): A list of file names to be avoided.Don't include any file extensions. Default is [].

        Returns:
        - list: A list of files to be segmented.
        """
        file_extensions = [".npz", ".jpg", ".png"]
        sample_filenames = get_sorted_files_with_extension(
            sample_direc, file_extensions
        )
        # filter out files whose filenames match any of the avoid_patterns
        sample_filenames = common.filter_files(sample_filenames, avoid_patterns)
        logger.info(f"files to seg: {sample_filenames}")
        return sample_filenames

    def create_model_data(self, directories: List[str], avoid_patterns: List[str] = []):
        translateoptions = "-of JPEG -co COMPRESS=JPEG -co TFW=YES -co QUALITY=100"
        for dir in directories:
            files = glob(os.path.join(dir, ".tif"))
            files = common.filter_files(files, avoid_patterns)
            logger.info(f"Translating tifs to jpgs {files}")
            common.gdal_translate_jpegs(files, translateoptions)

    def create_jpgs_for_tifs(
        self, directories: List[str], avoid_patterns: List["str"] = []
    ):
        """
        Creates JPEG files for TIF files in specified directories.

        Args:
            directories (List[str]): List of directory paths where TIF files are located.
            avoid_patterns (List[str], optional): List of file names to exclude. Defaults to [].

        Returns:
            None
        """

        translateoptions = "-of JPEG -co COMPRESS=JPEG -co TFW=YES -co QUALITY=100"
        for directory in tqdm.auto.tqdm(
            directories, desc="Convert .tif to .jpg", leave=False, unit_scale=True
        ):
            tif_files = self.get_tifs_missing_jpgs(directory, avoid_patterns)
            if len(tif_files) == 0:
                logger.info(
                    f"All tifs in directory '{directory}' have corresponding jpgs."
                )
                continue
            print(f"Converting tif files to jpgs in {directory}")
            logger.info(f"Translating tifs to jpgs {tif_files}")
            common.gdal_translate_jpegs(tif_files, translateoptions = translateoptions)

    def get_tifs_missing_jpgs(
        self, full_path: str, avoid_patterns: List[str] = []
    ) -> List[str]:
        """
        Returns a list of TIF files in `full_path` that are missing corresponding JPG files.

        Args:
            full_path (str): The path to the directory containing the TIF files.
            avoid_patterns (List[str]): A list of TIF file names to ignore.

        Returns:
            List[str]: A list of TIF file names that are missing corresponding JPG files.
        """
        tif_files = glob(os.path.join(full_path, "*.tif"))
        logger.info(f"all tif_files {tif_files}")
        tif_files = common.filter_files(tif_files, avoid_patterns)
        missing_jpgs = [
            file
            for file in tif_files
            if not os.path.exists(file.replace(".tif", ".jpg"))
        ]
        logger.info(f"Tiffs missing_jpgs {missing_jpgs}")
        return missing_jpgs

    def run_model(
        self,
        model_implementation: str,
        session_name: str,
        src_directory: str,
        model_name: str,
        use_GPU: str,
        use_otsu: bool,
        use_tta: bool,
    ):
        logger.info(f"ROI directory: {src_directory}")
        logger.info(f"session name: {session_name}")
        logger.info(f"model_name: {model_name}")
        logger.info(f"model_implementation: {model_implementation}")
        logger.info(f"use_GPU: {use_GPU}")
        logger.info(f"use_otsu: {use_otsu}")
        logger.info(f"use_tta: {use_tta}")

        self.download_model(model_implementation, model_name)
        weights_list = self.get_weights_list(model_implementation)

        # Load the model from the config files
        model, model_list, config_files, model_types = self.get_model(weights_list)
        metadatadict = self.get_metadatadict(weights_list, config_files, model_types)
        logger.info(f"metadatadict: {metadatadict}")

        # read classes from classes.txt from downloaded model directory
        downloaded_models_path = self.get_downloaded_models_dir()
        model_directory_path = os.path.abspath(
            os.path.join(downloaded_models_path, model_name)
        )
        class_path= os.path.join(model_directory_path,"classes.txt")
        classes = common.read_text_file(class_path)

        model_dict = {
            "sample_direc": None,
            "use_GPU": use_GPU,
            "implementation": model_implementation,
            "model_type": model_name,
            "otsu": False,
            "tta": False,
            "classes": classes,
        }
        
        session_path = common.create_directory(os.getcwd(), "sessions")
        session_dir = common.create_directory(session_path, session_name)

        # locate config files
        parent_directory = os.path.dirname(src_directory)
        config_gdf_path = common.find_config_json(parent_directory, r"config_gdf.*\.geojson")
        config_json_path = common.find_config_json(parent_directory, r"^config\.json$")

        year_dirs = common.get_matching_dirs(src_directory, pattern=r"^\d{4}$")
        year_tile_dirs = []
        # for each year
        for year_dir in tqdm.auto.tqdm(
            year_dirs, desc="Preparing data", leave=False, unit_scale=True
        ):
            original_merged_tif = os.path.join(year_dir,"merged_multispectral.tif")
            TARGET_SIZE = 768
            resampleAlg = 'mode'
            OVERLAP_PX = TARGET_SIZE//2
            tiles_path = common.create_directory(year_dir, "tiles")
            # run retile script with system command. retiles merged_multispectral.tif to have overlap
            cmd = f"python gdal_retile.py -r near -ot Byte -ps {TARGET_SIZE} {TARGET_SIZE} -overlap {OVERLAP_PX} -co 'tiled=YES' -targetDir {tiles_path} {original_merged_tif}"
            os.system(cmd)

            tif_files = glob(os.path.join(tiles_path,'*.tif'))
            kwargs = {
                'format': 'JPEG',
                'outputType': gdal.GDT_Byte
            }
            # create jpgs for new tifs
            common.gdal_translate_jpegs(tif_files,kwargs = kwargs)
            # delete tif files
            for file in tif_files:
                os.remove(file)

            if len(os.listdir(tiles_path)) == 0:
                continue
            year_tile_dirs.append(tiles_path) 

        # # create jpgs for all tifs that don't have one
        # self.create_jpgs_for_tifs(
        #     year_tile_dirs, avoid_patterns=[".*merged_multispectral.*"]
        # )
        # @todo get jpgs in each directory and add to a total to use for the progress bar
        logger.info(f"session directory: {session_dir}")

        for year_tile_dir in tqdm.auto.tqdm(
            year_tile_dirs, desc="Running models on each year", leave=False, unit_scale=True
        ):
            # make a model_settings.json
            model_year_dict = model_dict.copy()
            model_year_dict["sample_direc"] = year_tile_dir
            logger.info(f"model_year_dict: {model_year_dict}")
            year_name = os.path.basename(os.path.dirname(year_tile_dir))
            logger.info(f"year_name: {year_name}")
            session_year_path = common.create_directory(session_dir, year_name)
            logger.info(f"session_year_path: {session_year_path}")

            model_settings_path = os.path.join(session_year_path, "model_settings.json")
            common.write_to_json(model_settings_path, model_year_dict)
            # Compute the segmentation
            self.compute_segmentation(
                model_year_dict["sample_direc"],
                model_list,
                metadatadict,
                model_types,
                use_tta,
                use_otsu,
            )
            # move files from out dir to session directory under folder with year name
            session_year_path = common.create_directory(session_dir, year_name)

            # copy config files to session directory
            dst_file = os.path.join(session_year_path, "config_gdf.geojson")
            logger.info(f"dst_config_gdf: {dst_file}")
            shutil.copy(config_gdf_path, dst_file)
            dst_file = os.path.join(session_year_path, "config.json")
            logger.info(f"dst_config.json: {dst_file}")
            shutil.copy(config_json_path, dst_file)

            outputs_path = os.path.join(year_tile_dir,'out')
            logger.info(f"Moving from {outputs_path} files to {session_year_path}")
            if not os.path.exists(outputs_path):
                logger.info(f"No model outputs were generated for year {year_name}")
                print(f"No model outputs were generated for year {year_name}")
                continue

            
            # Remove "prob.png" files   
            _ = [os.remove(k) for k in glob(os.path.join(outputs_path,'*prob.png'))]

            # Remove "overlay.png" files ...
            _ = [os.remove(k) for k in glob(os.path.join(outputs_path,'*overlay.png'))]

            # Get imgs list
            predicition_pngs = sorted(glob(os.path.join(year_tile_dir, 'out', '*.png')))

            ## copy the xml files into the 'out' folder
            xml_files = sorted(glob(os.path.join(year_tile_dir, '*.xml')))
            for xml_file in xml_files:
                new_filename = xml_file.replace(year_tile_dir,outputs_path)
                if not os.path.isfile(new_filename):
                    shutil.copyfile(xml_file,new_filename)

            ## rename pngs
            for prediction in predicition_pngs:
                new_filename = prediction.replace('_predseg','')
                if not os.path.isfile(new_filename):
                    os.rename(prediction,new_filename)

            xml_files = sorted(glob(os.path.join(outputs_path, '*.xml')))
            ## rename xmls
            for xml_file in xml_files:
                new_filename = xml_file.replace('.jpg.aux.xml', '.png.aux.xml')
                if not os.path.isfile(new_filename):
                    os.rename(xml_file, new_filename)

            # @todo create_greyscale_tiff
            imgsToMosaic = common.create_greylabel_pngs(outputs_path)
            # xml files have  been renamed to have .png
            xml_files = sorted(glob(os.path.join(outputs_path, '*.xml')))
            print(f'{len(imgsToMosaic)} images to mosaic')
            # copy and name xmls
            for xml_file in xml_files:
                new_filename = xml_file.replace('.png','_res.png')
                if not os.path.isfile(new_filename):
                    shutil.copyfile(xml_file,new_filename)


            # create greyscale tif in session directory
            # generic vrt function
            resampleAlg = 'mode'
            outVRT = os.path.join(session_year_path, 'Mosaic_greyscale.vrt')
            outTIF = os.path.join(session_year_path, 'Mosaic_greyscale.tif')
            # First build vrt for geotiff output
            vrt_options = gdal.BuildVRTOptions(resampleAlg=resampleAlg)
            ds = gdal.BuildVRT(outVRT, imgsToMosaic, options=vrt_options)
            ds.FlushCache()
            ds = None
            # then build tiff
            ds = gdal.Translate(destName=outTIF, creationOptions=["NUM_THREADS=ALL_CPUS", "COMPRESS=LZW", "TILED=YES"], srcDS=outVRT)
            ds.FlushCache()
            ds = None

            # move segmentations to session directory
            dest =common.create_directory(session_year_path,'tiles')
            common.move_files_resurcively(src=year_tile_dir, dest=dest)
            # remove empty tiles directory
            if os.path.basename(year_tile_dir).lower() == "tiles":
                os.rmdir(year_tile_dir)



        # for year_dir in tqdm.auto.tqdm(
        #     year_dirs, desc="Creating tifs", leave=False, unit_scale=True
        # ):
        #     # move files from out dir to session directory under folder with year name
        #     year_name = os.path.basename(year_dir)
        #     outputs_path = os.path.join(src_directory, year_name, "out")
        #     session_year_path = common.create_directory(session_dir, year_name)
        #     logger.info(f"Moving from {outputs_path} files to {session_year_path}")
        #     if not os.path.exists(outputs_path):
        #         logger.info(f"No model outputs were generated for year {year_name}")
        #         print(f"No model outputs were generated for year {year_name}")
        #         continue

        #     common.move_files(outputs_path, session_year_path, delete_src=True)
        #     common.rename_files(
        #         session_year_path, "*.png", new_name="", replace_name="_predseg"
        #     )
        #     # copy the xml files associated with each model output
        #     xml_files = glob(os.path.join(year_dir, "*aux.xml"))
        #     common.copy_files(
        #         xml_files, session_year_path, avoid_patterns=[".*merged.*"]
        #     )
        #     # rename all the xml files
        #     common.rename_files(
        #         session_year_path, "*aux.xml", new_name=".png", replace_name=".jpg"
        #     )
        #     png_files = glob(os.path.join(session_year_path, "*png"))
        #     png_files = common.filter_files(png_files, [".*overlay.*"])
        #     common.gdal_translate_png_to_tiff(png_files, translateoptions="-of GTiff")
        #     logger.info(f"Done moving files for year : {session_year_path}")
        #     # create orthomoasic
        #     merged_multispectural = get_merged_multispectural(session_year_path)
        #     logger.info(f"merged_multispectural: {merged_multispectural}")

        #     # copy config files to session directory
        #     dst_file = os.path.join(session_year_path, "config_gdf.geojson")
        #     logger.info(f"dst_config_gdf: {dst_file}")
        #     shutil.copy(config_gdf_path, dst_file)
        #     dst_file = os.path.join(session_year_path, "config.json")
        #     logger.info(f"dst_config.json: {dst_file}")
        #     shutil.copy(config_json_path, dst_file)

    async def async_do_seg(self,
            semaphore,
            file_to_seg,
            model_list,
            metadatadict,
            model_type,
            sample_direc,
            NCLASSES,
            N_DATA_BANDS,
            TARGET_SIZE,
            TESTTIMEAUG ,
            WRITE_MODELMETADATA=False,
            OTSU_THRESHOLD=False,
            out_dir_name="out"):
        async with semaphore:
            logger.info(f"file_to_seg: {file_to_seg}")
            do_seg(
                file_to_seg,
                model_list,
                metadatadict,
                model_type,
                sample_direc,
                NCLASSES,
                N_DATA_BANDS,
                TARGET_SIZE,
                TESTTIMEAUG,
                WRITE_MODELMETADATA,
                OTSU_THRESHOLD,
                out_dir_name,
            )


    async def perform_segmentations(self,files_to_segment,
        model_list,
        metadatadict,
        model_types,
        sample_direc,
        NCLASSES,
        N_DATA_BANDS,
        TARGET_SIZE,
        TESTTIMEAUG ,
        WRITE_MODELMETADATA=False,
        OTSU_THRESHOLD=False,
        out_dir_name="out"):

        semaphore = asyncio.Semaphore(5)
        coroutines = []
        for file_to_seg in files_to_segment:
            coroutines.append(self.async_do_seg(
                semaphore,
                file_to_seg,
                model_list,
                metadatadict,
                model_types[0],
                sample_direc,
                NCLASSES,
                N_DATA_BANDS,
                TARGET_SIZE,
                TESTTIMEAUG,
                WRITE_MODELMETADATA,
                OTSU_THRESHOLD,
                out_dir_name,
            ))
        logger.info(f"coroutines: {coroutines}")
        await tqdm.asyncio.tqdm.gather(
                *coroutines,
                position=1,
                desc=f"Running Models",
            )

    @common.time_func
    def compute_segmentation(
        self,
        sample_direc: str,
        model_list: list,
        metadatadict: dict,
        model_types,
        use_tta: bool,
        use_otsu: bool,
    ):
 
        logger.info(f"Test Time Augmentation: {use_tta}")
        logger.info(f"Otsu Threshold: {use_otsu}")
        # Read in the image filenames as either .npz,.jpg, or .png
        files_to_segment = self.get_files_for_seg(sample_direc)
        logger.info(f"files_to_segment: {files_to_segment}")
        if model_types[0] != "segformer":
            ### mixed precision
            from tensorflow.keras import mixed_precision

            mixed_precision.set_global_policy("mixed_float16")
        # Compute the segmentation for each of the files
        # from src.seg2map.downloads import run_async_function
        # run_async_function(self.perform_segmentations,
        #                 files_to_segment=files_to_segment,
        #                 model_list=model_list,
        #                 metadatadict=metadatadict,
        #                 model_types=model_types,
        #                 sample_direc=sample_direc,
        #                 NCLASSES=self.NCLASSES,
        #                 N_DATA_BANDS=self.N_DATA_BANDS,
        #                 TARGET_SIZE=self.TARGET_SIZE,
        #                 TESTTIMEAUG=use_tta,
        #                 WRITE_MODELMETADATA=False,
        #                 OTSU_THRESHOLD=use_otsu,
        #                 out_dir_name="out",
        #                         )
        for file_to_seg in tqdm.auto.tqdm(files_to_segment):
            do_seg(
                file_to_seg,
                model_list,
                metadatadict,
                model_types[0],
                sample_direc=sample_direc,
                NCLASSES=self.NCLASSES,
                N_DATA_BANDS=self.N_DATA_BANDS,
                TARGET_SIZE=self.TARGET_SIZE,
                TESTTIMEAUG=use_tta,
                WRITE_MODELMETADATA=False,
                OTSU_THRESHOLD=use_otsu,
                out_dir_name="out",
                profile="meta",
            )

    @common.time_func
    def get_model(self, weights_list: list):
        model_list = []
        config_files = []
        model_types = []
        if weights_list == []:
            raise Exception("No Model Info Passed")
        for weights in weights_list:
            # "fullmodel" is for serving on zoo they are smaller and more portable between systems than traditional h5 files
            # gym makes a h5 file, then you use gym to make a "fullmodel" version then zoo can read "fullmodel" version
            configfile = (
                weights.replace(".h5", ".json").replace("weights", "config").strip()
            )
            if "fullmodel" in configfile:
                configfile = configfile.replace("_fullmodel", "").strip()
            with open(configfile) as f:
                config = json.load(f)
            self.TARGET_SIZE = config.get("TARGET_SIZE")
            MODEL = config.get("MODEL")
            self.NCLASSES = config.get("NCLASSES")
            KERNEL = config.get("KERNEL")
            STRIDE = config.get("STRIDE")
            FILTERS = config.get("FILTERS")
            self.N_DATA_BANDS = config.get("N_DATA_BANDS")
            DROPOUT = config.get("DROPOUT")
            DROPOUT_CHANGE_PER_LAYER = config.get("DROPOUT_CHANGE_PER_LAYER")
            DROPOUT_TYPE = config.get("DROPOUT_TYPE")
            USE_DROPOUT_ON_UPSAMPLING = config.get("USE_DROPOUT_ON_UPSAMPLING")
            DO_TRAIN = config.get("DO_TRAIN")
            LOSS = config.get("LOSS")
            PATIENCE = config.get("PATIENCE")
            MAX_EPOCHS = config.get("MAX_EPOCHS")
            VALIDATION_SPLIT = config.get("VALIDATION_SPLIT")
            RAMPUP_EPOCHS = config.get("RAMPUP_EPOCHS")
            SUSTAIN_EPOCHS = config.get("SUSTAIN_EPOCHS")
            EXP_DECAY = config.get("EXP_DECAY")
            START_LR = config.get("START_LR")
            MIN_LR = config.get("MIN_LR")
            MAX_LR = config.get("MAX_LR")
            FILTER_VALUE = config.get("FILTER_VALUE")
            DOPLOT = config.get("DOPLOT")
            ROOT_STRING = config.get("ROOT_STRING")
            USEMASK = config.get("USEMASK")
            AUG_ROT = config.get("AUG_ROT")
            AUG_ZOOM = config.get("AUG_ZOOM")
            AUG_WIDTHSHIFT = config.get("AUG_WIDTHSHIFT")
            AUG_HEIGHTSHIFT = config.get("AUG_HEIGHTSHIFT")
            AUG_HFLIP = config.get("AUG_HFLIP")
            AUG_VFLIP = config.get("AUG_VFLIP")
            AUG_LOOPS = config.get("AUG_LOOPS")
            AUG_COPIES = config.get("AUG_COPIES")
            REMAP_CLASSES = config.get("REMAP_CLASSES")

            try:
                model = tf.keras.models.load_model(weights)
                #  nclasses=NCLASSES, may have to replace nclasses with NCLASSES
            except BaseException:
                if MODEL == "resunet":
                    model = custom_resunet(
                        (self.TARGET_SIZE[0], self.TARGET_SIZE[1], self.N_DATA_BANDS),
                        FILTERS,
                        nclasses=[
                            self.NCLASSES + 1 if self.NCLASSES == 1 else self.NCLASSES
                        ][0],
                        kernel_size=(KERNEL, KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,  # 0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                        dropout_type=DROPOUT_TYPE,  # "standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                    )
                elif MODEL == "unet":
                    model = custom_unet(
                        (self.TARGET_SIZE[0], self.TARGET_SIZE[1], self.N_DATA_BANDS),
                        FILTERS,
                        nclasses=[
                            self.NCLASSES + 1 if self.NCLASSES == 1 else self.NCLASSES
                        ][0],
                        kernel_size=(KERNEL, KERNEL),
                        strides=STRIDE,
                        dropout=DROPOUT,  # 0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                        dropout_type=DROPOUT_TYPE,  # "standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                    )

                elif MODEL == "simple_resunet":
                    # num_filters = 8 # initial filters
                    model = simple_resunet(
                        (self.TARGET_SIZE[0], self.TARGET_SIZE[1], self.N_DATA_BANDS),
                        kernel=(2, 2),
                        num_classes=[
                            self.NCLASSES + 1 if self.NCLASSES == 1 else self.NCLASSES
                        ][0],
                        activation="relu",
                        use_batch_norm=True,
                        dropout=DROPOUT,  # 0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                        dropout_type=DROPOUT_TYPE,  # "standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                        filters=FILTERS,  # 8,
                        num_layers=4,
                        strides=(1, 1),
                    )
                    # 346,564
                elif MODEL == "simple_unet":
                    model = simple_unet(
                        (self.TARGET_SIZE[0], self.TARGET_SIZE[1], self.N_DATA_BANDS),
                        kernel=(2, 2),
                        nclasses=self.NCLASSES,
                        activation="relu",
                        use_batch_norm=True,
                        dropout=DROPOUT,  # 0.1,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                        dropout_type=DROPOUT_TYPE,  # "standard",
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                        filters=FILTERS,  # 8,
                        num_layers=4,
                        strides=(1, 1),
                    )
                elif MODEL == "satunet":
                    model = simple_satunet(
                        (self.TARGET_SIZE[0], self.TARGET_SIZE[1], self.N_DATA_BANDS),
                        kernel=(2, 2),
                        num_classes=self.NCLASSES,  # [NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
                        activation="relu",
                        use_batch_norm=True,
                        dropout=DROPOUT,
                        dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,
                        dropout_type=DROPOUT_TYPE,
                        use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,
                        filters=FILTERS,
                        num_layers=4,
                        strides=(1, 1),
                    )
                elif MODEL == "segformer":
                    id2label = {}
                    for k in range(self.NCLASSES):
                        id2label[k] = str(k)
                    model = segformer(id2label, num_classes=self.NCLASSES)
                    model.compile(optimizer="adam")
                # 242,812
                else:
                    raise Exception(
                        f"An unknown model type {MODEL} was received. Please select a valid model.\n \
                        Model must be one of 'unet', 'resunet', 'segformer', or 'satunet'"
                    )

                # Load in the custom loss function from doodleverse_utils
                model.compile(
                    optimizer="adam", loss=dice_coef_loss(self.NCLASSES)
                )  # , metrics = [iou_multi(self.NCLASSESNCLASSES), dice_multi(self.NCLASSESNCLASSES)])

                model.load_weights(weights)

            model_types.append(MODEL)
            model_list.append(model)
            config_files.append(configfile)

        return model, model_list, config_files, model_types

    @common.time_func
    def get_metadatadict(
        self, weights_list: list, config_files: list, model_types: list
    ):
        metadatadict = {}
        metadatadict["model_weights"] = weights_list
        metadatadict["config_files"] = config_files
        metadatadict["model_types"] = model_types
        return metadatadict

    @common.time_func
    def get_weights_list(self, model_choice: str = "ENSEMBLE"):
        """Returns of the weights files(.h5) within weights_direc"""
        if model_choice == "ENSEMBLE":
            weights_list = glob(self.weights_direc + os.sep + "*.h5")
            logger.info(f"ENSEMBLE: weights_list: {weights_list}")
            logger.info(
                f"ENSEMBLE: {len(weights_list)} sets of model weights were found "
            )
            return weights_list
        elif model_choice == "BEST":
            # read model name (fullmodel.h5) from BEST_MODEL.txt
            with open(self.weights_direc + os.sep + "BEST_MODEL.txt") as f:
                model_name = f.readlines()
            weights_list = [self.weights_direc + os.sep + model_name[0]]
            logger.info(f"BEST: weights_list: {weights_list}")
            logger.info(f"BEST: {len(weights_list)} sets of model weights were found ")
            return weights_list

    def get_downloaded_models_dir(self) -> str:
        """returns full path to downloaded_models directory and
        if downloaded_models directory does not exist then it is created

        Returns:
            str: full path to downloaded_models directory
        """
        # directory to hold downloaded models from Zenodo
        script_dir = os.path.dirname(os.path.abspath(__file__))
        downloaded_models_path = os.path.abspath(
            os.path.join(script_dir, "downloaded_models")
        )
        if not os.path.exists(downloaded_models_path):
            os.mkdir(downloaded_models_path)
        logger.info(f"downloaded_models_path: {downloaded_models_path}")
        return downloaded_models_path

    def download_model(self, model_choice: str, dataset_id: str) -> None:
        """downloads model specified by zenodo id in dataset_id.

        Downloads best model is model_choice = 'BEST' or all models in
        zenodo release if model_choice = 'ENSEMBLE'

        Args:
            model_choice (str): 'BEST' or 'ENSEMBLE'
            dataset_id (str): name of model followed by underscore zenodo_id'name_of_model_zenodoid'
        """
        # Extract the Zenodo ID from the dataset ID
        zenodo_id = dataset_id.split("_")[-1]

        # Construct the URL for the Zenodo release
        root_url = f"https://zenodo.org/api/records/{zenodo_id}"

        # Retrieve the JSON data for the Zenodo release
        response = requests.get(root_url)
        response.raise_for_status()
        json_content = response.json()
        files = json_content["files"]

        # Create a directory to hold the downloaded models
        downloaded_models_path = self.get_downloaded_models_dir()
        self.weights_direc = os.path.abspath(
            os.path.join(downloaded_models_path, dataset_id)
        )
        os.makedirs(self.weights_direc, exist_ok=True)
        logger.info(f"self.weights_direc:{self.weights_direc}")
        print(f"\n Model located at: {self.weights_direc}")

        models_json_dict = {}
        if model_choice.upper() == "BEST":
            best_model_json = next((f for f in files if f["key"] == "BEST_MODEL.txt"), None)
            if best_model_json is None:
                raise ValueError(f"Cannot find BEST_MODEL.txt at {root_url}")
            logger.info(f"BEST_MODEL.txt: {best_model_json}")

            best_model_path = os.path.join(self.weights_direc, "BEST_MODEL.txt")
            logger.info(f"best_model_path : {best_model_path}")

            # if BEST_MODEL.txt file not exist download it
            if not os.path.isfile(best_model_path):
                download_url(best_model_json["links"]["self"], best_model_path)
        
            # read filename of the best model in BEST_MODEL.txt
            with open(best_model_path, "r") as f:
                filename = f.read().strip()

            # make sure the best model file is online and
            model_json = next((f for f in files if f["key"] == filename), None)
            if model_json is None:
                raise ValueError(f"Cannot find file '{filename}' in {root_url}. Raise an Issue on Github.")
            logger.info(f"Model: {model_json}")

            # check if json and h5 file in BEST_MODEL.txt exist
            # model_json = [f for f in files if f["key"] == filename][0]
            # path to save model
            model_path = os.path.join(self.weights_direc, filename)
            logger.info(f"model_path for BEST_MODEL.txt: {model_path}")

            # path to save file and json data associated with file saved to dict
            models_json_dict[model_path] = model_json["links"]["self"]
        elif model_choice.upper() == "ENSEMBLE":
            # get list of all models
            all_models = [f for f in files if f["key"].endswith(".h5")]
            if not all_models:
                raise ValueError(f"No .h5 files found at {root_url}")
            logger.info(f"All models: {all_models}")

            # check if all h5 files in files are in self.weights_direc
            for model_json in all_models:
                model_path = os.path.join(self.weights_direc, model_json["links"]["self"].split("/")[-1])
                if not os.path.isfile(model_path):
                    download_url(model_json["links"]["self"], model_path)

                logger.info(f"ENSEMBLE: model_path: {model_path}")
                # save url to download ensemble models. each url is identified by path to save model
                models_json_dict[model_path] = model_json["links"]["self"]
        else:
            raise ValueError(f"Invalid model_choice '{model_choice}'")

        # make sure classes.txt file is downloaded
        filename="classes.txt"
        classes_file_json = next((f for f in files if f["key"] == filename), None)
        if classes_file_json is None:
            raise ValueError(f"Cannot find {filename} at {root_url}")
        
        file_path = os.path.join(self.weights_direc, filename)
        if not os.path.isfile(file_path):
            models_json_dict[file_path] = classes_file_json["links"]["self"]

        logger.info(f"models_json_dict: {models_json_dict}")
        # filter through the files and keep only the files that haven't been downloaded
        url_dict = get_url_dict_to_download(models_json_dict)
        logger.info(f"URLs to download: {url_dict}")
        # if any files are not found locally download them asynchronous
        if url_dict != {}:
            run_async_download(url_dict)
