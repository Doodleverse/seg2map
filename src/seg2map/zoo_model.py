from typing import Dict, List, Optional, Set
from typing import Dict, Any
import requests
import logging
import os
import shutil
import json
from pathlib import Path
from glob import glob

from seg2map import common
from seg2map import downloads
from seg2map import sessions
from seg2map import map_functions

from skimage.io import imsave
import tqdm
import numpy as np
from osgeo import gdal
import tqdm.asyncio
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

def process_roi(roi_path: str, model_dict: dict) -> Dict:
    """
    Process a Region of Interest (ROI) directory and return a dictionary containing the
    processed data for each year within the ROI.

    Args:
        roi_path (str): The path to the ROI directory.
        model_dict (dict): A dictionary containing model-specific configuration data.

    Returns:
        Dict: A dictionary where keys are the years (e.g., '2010', '2011', ...) and
              values are dictionaries containing model configuration data specific to
              that year, including the path to the preprocessed data directory for
              that year. For example:

                  {
                      '2010': {"model_type": "OpenEarthNet_RGB_9class_7576894",, 'sample_direc': '/path/to/roi/2010/data', ...},
                      '2011': {"model_type": "OpenEarthNet_RGB_9class_7576894",, 'sample_direc': '/path/to/roi/2011/data', ...},
                      ...
                  }

    Example usage:
        roi_path = '/path/to/roi_directory'
        model_dict = {'model_param1': value1, 'model_param2': value2, ...}
        model_data_per_year = process_roi(roi_path, model_dict)
    """
    model_data_per_year = {}
    multiband_path = os.path.join(roi_path, "multiband")
    year_dirs = common.get_matching_dirs(multiband_path, pattern=r"^\d{4}$")
    
    for year_dir in tqdm.auto.tqdm(year_dirs, desc="Preparing data", leave=False, unit_scale=True):
        model_year_dict = process_year_directory(year_dir, model_dict)
        
        if model_year_dict:
            year_key = os.path.basename(year_dir)
            model_data_per_year[year_key] = model_year_dict
    
    return model_data_per_year

def process_year_directory(year_dir: str, model_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a year's directory of multispectral images by creating overlapping tiles,
    and update the model dictionary with the new sample directory.

    Args:
        year_dir (str): Path to the year's directory containing multispectral images.
        model_dict (Dict[str, Any]): The original model dictionary.

    Returns:
        Dict[str, Any]: The updated model dictionary with the new sample directory, or None if input directory is empty.

    """
    if len(os.listdir(year_dir)) == 0:
        logger.warning(f"len(os.listdir({year_dir})) == 0")
        return {}
    # everything after this assumes that the directory is not empty
    # Construct the file path to the original merged multispectral image
    original_merged_tif = os.path.join(year_dir, "merged_multispectral.tif")
    # Create a new "tiles" directory within the year_dir
    tiles_path = common.create_directory(year_dir, "tiles")
    # Generate overlapping tiles from the original merged multispectral image and get the new tiles path
    tiles_path = create_overlapping_tiles(original_merged_tif, tiles_path)
    # Check if the returned tiles_path is empty or None, log a warning and return an empty dictionary if true
    if not tiles_path:
        logger.warning(f"Overlap operation failed for {year_dir} for merged tif: {original_merged_tif}")
        return {}
    # Create a copy of the model dictionary and update the sample directory with the new tiles path
    model_year_dict = model_dict.copy()
    model_year_dict["sample_direc"] = tiles_path
    return model_year_dict

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

def save_greyscale_label_as_png(npz_location: str) -> str:
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

def generate_greyscale_label_pngs(full_path: str) -> List[str]:
    """
    Given a directory path, finds all .npz files in the directory, writes the 'grey_label' array of each .npz
    file to a corresponding PNG file, and returns a list of the paths of the written PNG files.

    Parameters:
    full_path (str): The path of the directory to search for .npz files.

    Returns:
    List[str]: A list of the paths of the written PNG files.
    """
    png_files = []
    logger.info(f"full_path: {full_path}")
    npzs = sorted(glob(os.path.join(full_path, "*.npz")))
    logger.info(f"npzs: {npzs}")
    for npz in npzs:
        png_files.append(save_greyscale_label_as_png(npz))
    return png_files

def rename_xmls(src: str, old_name: str, new_name: str) -> None:
    """
    Renames all .xml files in the source directory by replacing 'old_name' with 'new_name'.
    
    Parameters:
    src (str): The path of the source directory containing .xml files.
    old_name (str): The old substring to be replaced in the .xml filenames.
    new_name (str): The new substring to replace the 'old_name' in the .xml filenames.
    """
    xml_files = sorted(Path(src).glob("*.xml"))
    for xml_file in xml_files:
        new_filename = str(xml_file).replace(old_name, new_name)
        if not Path(new_filename).is_file():
            xml_file.rename(new_filename)

def copy_xmls(src: str, dst: str) -> None:
    """
    Copies all .xml files from the source directory to the destination directory.

    Parameters:
    src (str): The path of the source directory containing .xml files.
    dst (str): The path of the destination directory where .xml files will be copied.
    """
    xml_files = sorted(Path(src).glob("*.xml"))
    for xml_file in xml_files:
        new_filename = str(xml_file).replace(src, dst)
        if not Path(new_filename).is_file():
            shutil.copyfile(str(xml_file), new_filename)


def remove_unused_files(outputs_path: str) -> None:
    """
    Removes unused files (e.g., 'prob.png' and 'overlay.png') from the specified directory.

    Parameters:
    outputs_path (str): The path of the directory containing unused files to be removed.
    """
    for file_extension in ["*prob.png", "*overlay.png"]:
        for unused_file in Path(outputs_path).glob(file_extension):
            unused_file.unlink()


def rename_predictions(predictions_location: str) -> None:
    """
    Renames prediction PNG files in the specified directory by removing the '_predseg' suffix.

    Parameters:
    predictions_location (str): The path of the directory containing prediction PNG files.
    """
    # find predictions and rename
    # Get imgs list
    predicition_pngs = sorted(glob(os.path.join(predictions_location, "*.png")))
    # rename pngs
    for prediction in predicition_pngs:
        new_filename = prediction.replace("_predseg", "")
        if not os.path.isfile(new_filename):
            os.rename(prediction, new_filename)


def make_greyscale_tif(tiles_location: str, tif_location: str) -> str:
    """
    Converts a set of tiled images into a greyscale mosaic TIFF file.

    Args:
    tiles_location (str): Path to the directory containing the input tiled images.
    tif_location (str): Path to the directory where the output TIFF file will be saved.

    Returns:
    str: The path to the generated greyscale mosaic TIFF file.

    Raises:
    None.
    """
    logger.info(f"tiles_location: {tiles_location}")
    logger.info(f"tif_location: {tif_location}")
    outputs_path = os.path.join(tiles_location, "out")
    logger.info(f"outputs_path: {outputs_path}")
    # copy xmls to the out folder
    copy_xmls(tiles_location, outputs_path)
    rename_xmls(outputs_path, ".jpg.aux.xml", ".png.aux.xml")
    rename_predictions(outputs_path)
    # create pngs from the npz files (segmentations)
    imgsToMosaic = generate_greyscale_label_pngs(outputs_path)
    if len(imgsToMosaic) == 0:
        logger.warning("No segmented images were found")
        return ""
    # rename the segmented images
    rename_xmls(outputs_path, ".png", "_res.png")
    outVRT = os.path.join(tif_location, "Mosaic_greyscale.vrt")
    outTIF = os.path.join(tif_location, "Mosaic_greyscale.tif")
    common.build_vrt(outVRT, imgsToMosaic, resampleAlg="mode")
    # create greyscale tiff
    common.build_tiff(outTIF, outVRT)
    return outTIF

def gdal_retile(tif_path: str, tiles_path: str, OVERLAP_PX: Optional[int] = None, TARGET_SIZE: int = 768)->str:
    """
    Retiles a merged GeoTIFF file into smaller overlapping tiles of the specified size.

    Args:
        tif_path (str): Path to the input GeoTIFF file.
        tiles_path (str): Output directory where generated tiles will be saved.
        OVERLAP_PX (int, optional): Number of pixels to overlap between tiles. Defaults to None, which calculates the overlap as half of the target tile size.
        TARGET_SIZE (int, optional): Target size of each tile in pixels (width and height). Defaults to 768.

    Returns:
        str: The output directory where the generated tiles are saved.

    Raises:
        Exception: If an error occurs during the retile process.
    """
    try:
        # retile merged tif and create tiles with overlap
        if not OVERLAP_PX:
            OVERLAP_PX = TARGET_SIZE // 2
        # run retile script with system command. retiles merged_multispectral.tif to become many tif files that overlap each other
        # cmd = f"python gdal_retile.py -r near -ot Byte -ps {TARGET_SIZE} {TARGET_SIZE} -overlap {OVERLAP_PX} -co 'tiled=YES' -targetDir {tiles_path} {tif_path}"
        # cmd = f"python gdal_retile.py -r near -ot Byte -ps {TARGET_SIZE} {TARGET_SIZE} -overlap {OVERLAP_PX} -co 'TILED=YES' -co 'BLOCKXSIZE={TARGET_SIZE}' -co 'BLOCKYSIZE={TARGET_SIZE}' -targetDir {tiles_path} {tif_path}"
        cmd = f"python gdal_retile.py -r near -ot Byte -ps {TARGET_SIZE} {TARGET_SIZE} -overlap {OVERLAP_PX} -targetDir {tiles_path} {tif_path}"
        logger.info(f"overlap command: {cmd}")
        os.system(cmd)
        # return location of tiles that were created
    except Exception as e:
        logger.error(f"{e}\ntif_path: {tif_path}\n tiles_path: {tiles_path}\n OVERLAP_PX: {OVERLAP_PX}\n TARGET_SIZE: {TARGET_SIZE}")
        raise e
    return tiles_path

def create_overlapping_tiles(
    tif_path: str, tiles_path: str, OVERLAP_PX: int = None, TARGET_SIZE: int = 768
)->str:
    """
    Creates overlapping tiles of a GeoTIFF file and converts them to JPEG format.

    Args:
        tif_path (str): Path to the input GeoTIFF file.
        tiles_path (str): Output directory where generated tiles will be saved.
        OVERLAP_PX (int, optional): Number of pixels to overlap between tiles. Defaults to None, which calculates the overlap as half of the target tile size.
        TARGET_SIZE (int, optional): Target size of each tile in pixels (width and height). Defaults to 768.

    Returns:
        str: The output directory where the generated overlapping JPEG tiles are saved.

    Raises:
        FileNotFoundError: If the input GeoTIFF file is not found.
    """
    # this means the directory is empty
    if not os.path.exists(tif_path):
        return ""
    # retile merged tif to create tifs with overlap
    tiles_location = gdal_retile(tif_path, tiles_path, OVERLAP_PX, TARGET_SIZE)
    tif_files = glob(os.path.join(tiles_location, "*.tif"))
    if not tif_files:
        logger.error(f"Overlap failed. No tif files were found in {tiles_location}")
        raise Exception(f"Overlap failed. No tif files were found in {tiles_location}")
        return ""
    
    jpeg_kwargs = {"format": "JPEG", "outputType": gdal.GDT_Byte}
    # Convert tif files to jpgs
    jpg_files = common.gdal_translate_jpegs(tif_files, kwargs=jpeg_kwargs)
    if not jpg_files:
        logger.error(f"Overlap failed. No jpg files were found in {tiles_location}")
        raise Exception(f"Overlap failed. No tif files were found in {tiles_location}")
        return ""
    # delete tif files to save space
    for file in tif_files:
        os.remove(file)
    # return location of overlapping jpgs that were created
    return tiles_path


def download_url(
    url: str, save_path: str, progress_bar_name: str = "", chunk_size: int = 1024
):
    """Downloads the model from the given url to the save_path location.
    Args:
        url (str): url to model to download
        save_path (str): directory to save model
        chunk_size (int, optional):  Defaults to 1024.
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
                    desc=progress_bar_name,
                    initial=0,
                    ascii=True,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        fd.write(chunk)
                        pbar.update(len(chunk))


def get_zenodo_release(zenodo_id: str) -> dict:
    """
    Retrieves the JSON data for the Zenodo release with the given ID.
    """
    root_url = f"https://zenodo.org/api/records/{zenodo_id}"
    response = requests.get(root_url)
    response.raise_for_status()
    return response.json()


def get_files_to_download(
    available_files: List[dict], filenames: List[str], model_id: str, model_path: str
) -> dict:
    """Constructs a dictionary of file paths and their corresponding download links, based on the available files and a list of desired filenames.

    Args:
    - available_files: A list of dictionaries representing the metadata of available files, including the file key and download links.
    - filenames: A list of strings representing the desired filenames.
    - model_id: A string representing the ID of the model being downloaded.
    - model_path: A string representing the path to the directory where the files will be downloaded.

    Returns:
    A dictionary with file paths as keys and their corresponding download links as values.
    Raises a ValueError if any of the desired filenames are not available in the available_files list.
    """
    # make sure classes.txt file is downloaded
    if isinstance(filenames, str):
        filenames = [filenames]
    url_dict = {}
    for filename in filenames:
        response = next((f for f in available_files if f["key"] == filename), None)
        if response is None:
            raise ValueError(f"Cannot find {filename} at {model_id}")
        link = response["links"]["self"]
        file_path = os.path.join(model_path, filename)
        url_dict[file_path] = link
    return url_dict


def check_if_files_exist(files_dict: dict) -> dict:
    """Checks if each file in a given dictionary of file paths and download links already exists in the local filesystem.

    Args:
    - files_dict: A dictionary with file paths as keys and their corresponding download links as values.

    Returns:
    A dictionary with file paths as keys and their corresponding download links as values, for any files that do not exist in the local filesystem.
    """
    url_dict = {}
    for save_path, link in files_dict.items():
        if not os.path.isfile(save_path):
            url_dict[save_path] = link
    return url_dict


class ZooModel:
    def __init__(self):
        self.model = None
        self.weights_directory = ""
        self.model_dict = {}

    def get_model_directory(self, model_id: str):
        # Create a directory to hold the downloaded models
        downloaded_models_path = self.get_downloaded_models_dir()
        model_directory = common.create_directory(downloaded_models_path, model_id)
        return model_directory

    def download_model(
        self, model_choice: str, model_id: str, model_path: str = None
    ) -> None:
        """downloads model specified by zenodo id in model_id.

        Downloads best model is model_choice = 'BEST' or all models in
        zenodo release if model_choice = 'ENSEMBLE'

        Args:
            model_choice (str): 'BEST' or 'ENSEMBLE'
            model_id (str): name of model followed by underscore zenodo_id ex.'orthoCT_RGB_2class_7574784'
            model_path (str): path to directory to save the downloaded files to
        """
        # Extract the Zenodo ID from the dataset ID
        zenodo_id = model_id.split("_")[-1]
        # get list of files available in zenodo release
        json_content = get_zenodo_release(zenodo_id)
        available_files = json_content["files"]

        # Download the best model if best or all models if ensemble
        if model_choice.upper() == "BEST":
            self.download_best(available_files, model_path, model_id)
        elif model_choice.upper() == "ENSEMBLE":
            self.download_ensemble(available_files, model_path, model_id)

    def download_best(
        self, available_files: List[dict], model_path: str, model_id: str
    ):
        """
        Downloads the best model file and its corresponding JSON and classes.txt files from the given list of available files.

        Args:
            available_files (list): A list of files available to download.
            model_path (str): The local directory where the downloaded files will be stored.
            model_id (str): The ID of the model being downloaded.

        Raises:
            ValueError: If BEST_MODEL.txt file is not found in the given model_id.

        Returns:
            None
        """
        download_dict = {}
        # download best_model.txt and read the name of the best model
        best_model_json = next(
            (f for f in available_files if f["key"] == "BEST_MODEL.txt"), None
        )
        if best_model_json is None:
            raise ValueError(f"Cannot find BEST_MODEL.txt in {model_id}")
        # download best model file to check if it exists
        BEST_MODEL_txt_path = os.path.join(model_path, "BEST_MODEL.txt")
        logger.info(f"model_path for BEST_MODEL.txt: {BEST_MODEL_txt_path}")
        # if best BEST_MODEL.txt file not exist then download it
        if not os.path.isfile(BEST_MODEL_txt_path):
            download_url(
                best_model_json["links"]["self"],
                BEST_MODEL_txt_path,
                progress_bar_name="Downloading best_model.txt",
            )

        with open(BEST_MODEL_txt_path, "r") as f:
            best_model_filename = f.read().strip()
        # get the json data of the best model _fullmodel.h5 file
        best_json_filename = best_model_filename.replace("_fullmodel.h5", ".json")
        best_modelcard_filename = best_model_filename.replace(
            "_fullmodel.h5", "_modelcard.json"
        )

        # download best model files(.h5, .json) file and classes.txt
        download_filenames = [
            "classes.txt",
            best_json_filename,
            best_model_filename,
            best_modelcard_filename,
        ]
        download_dict.update(
            get_files_to_download(
                available_files, download_filenames, model_id, model_path
            )
        )

        download_dict = check_if_files_exist(download_dict)
        # download the files that don't exist
        logger.info(f"URLs to download: {download_dict}")
        # if any files are not found locally download them asynchronous
        if download_dict != {}:
            downloads.run_async_function(
                downloads.async_download_url_dict, url_dict=download_dict
            )

    def download_ensemble(
        self, available_files: List[dict], model_path: str, model_id: str
    ):
        """
        Downloads all the model files and their corresponding JSON and classes.txt files from the given list of available files, for an ensemble model.

        Args:
            available_files (list): A list of files available to download.
            model_path (str): The local directory where the downloaded files will be stored.
            model_id (str): The ID of the model being downloaded.

        Raises:
            Exception: If no .h5 files or corresponding .json files are found in the given model_id.

        Returns:
            None
        """
        download_dict = {}
        # get json and models
        all_models_reponses = [f for f in available_files if f["key"].endswith(".h5")]
        all_model_names = [f["key"] for f in all_models_reponses]
        json_file_names = [
            model_name.replace("_fullmodel.h5", ".json")
            for model_name in all_model_names
        ]
        modelcard_file_names = [
            model_name.replace("_fullmodel.h5", "_modelcard.json")
            for model_name in all_model_names
        ]
        all_json_reponses = []
        for available_file in available_files:
            if available_file["key"] in json_file_names + modelcard_file_names:
                all_json_reponses.append(available_file)
        if len(all_models_reponses) == 0:
            raise Exception(f"Cannot find any .h5 files at {model_id}")
        if len(all_json_reponses) == 0:
            raise Exception(
                f"Cannot find corresponding .json files for .h5 files at {model_id}"
            )

        logger.info(f"all_models_reponses : {all_models_reponses }")
        logger.info(f"all_json_reponses : {all_json_reponses }")
        for response in all_models_reponses + all_json_reponses:
            # get the link of the best model
            link = response["links"]["self"]
            filename = response["key"]
            filepath = os.path.join(model_path, filename)
            download_dict[filepath] = link
        # download classes.txt file
        download_dict.update(
            get_files_to_download(
                available_files, ["classes.txt"], model_id, model_path
            )
        )
        download_dict = check_if_files_exist(download_dict)
        # download the files that don't exist
        logger.info(f"URLs to download: {download_dict}")
        # if any files are not found locally download them asynchronous
        if download_dict != {}:
            downloads.run_async_function(
                downloads.async_download_url_dict, url_dict=download_dict
            )

    def get_model(self, weights_list: list):
        if not weights_list:
            raise Exception("No Model Info Passed")
    
        model_list = []
        config_files = []
        model_types = []
    
        for weights_path in weights_list:
            weights_path = Path(weights_path)
            config_path = weights_path.with_name(weights_path.stem.replace("weights", "config")).with_suffix(".json")
    
            if "fullmodel" in config_path.name:
                config_path = config_path.with_name(config_path.stem.replace("_fullmodel", "") + config_path.suffix)
    
            with config_path.open() as f:
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

            try:
                model = tf.keras.models.load_model(str(weights_path))
            except Exception:
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
                    id2label = {k: str(k) for k in range(self.NCLASSES)}
                    model = segformer(id2label, num_classes=self.NCLASSES)
                    model.compile(optimizer="adam")
                else:
                    raise Exception(
                        f"An unknown model type {MODEL} was received. Please select a valid model.\n \
                        Model must be one of 'unet', 'resunet', 'segformer', or 'satunet'"
                    )
    
                model.compile(optimizer="adam", loss=dice_coef_loss(self.NCLASSES))
                try:
                    model.load_weights(str(weights_path))
                except Exception as e:
                    logger.info(f"Error loading the weights files {weights_path}\n Error {e}")
                    raise Exception(f"Error loading the weights files {weights_path}.\n This is likely due to a download error, try deleting the model from downloaded_models and try again.\n Error {e}")
    
            model_types.append(MODEL)
            model_list.append(model)
            config_files.append(str(config_path))
    
        return model, model_list, config_files, model_types


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


    def preprocess_data(
        self, src_directory: str, model_dict: dict, session: sessions.Session
    )-> dict:
        """
        Preprocesses the data for regions of interest (ROIs) given a source directory
        and a dictionary containing model information.
        
        Args:
        - src_directory (str): Path to the source directory containing the ROIs.
        - model_dict (dict): Dictionary containing key-value pairs related to the model.
        - session (Session): A session object which holds other essential data.

        Returns:
        - preprocessed_data (dict): A dictionary containing preprocessed data for each
          ROI for a specific year, including the path to the ROI.

        """
        # load year directories for each ROI
        roi_dict = common.get_subdirectories_with_ids(src_directory)
        logger.info(f"roi_dict: {roi_dict}")
        
        session.roi_ids = list(roi_dict.keys())
        # create dictionary to run on model on for each year in each ROI
        preprocessed_data = {}
        for roi_id, roi_path in roi_dict.items():
            # for each year create overlapping tiles and save location of tiles
            model_data_per_year = process_roi(roi_path, model_dict)
            if model_data_per_year:
                preprocessed_data[roi_id] = model_data_per_year
                preprocessed_data[roi_id]["roi_path"] = roi_path
                
        logger.info(f"preprocessed_data: {preprocessed_data}")

        return preprocessed_data

    def compute_segmentation(
        self,
        preprocessed_data: dict,
    ):
        """
        Computes the segmentation of the input multispectral images using the preprocessed data.

        Args:
            preprocessed_data (dict): A dictionary containing the preprocessed data that's in a format that's
            ready for segmentation.

        Returns:
            None.

        Raises:
            None.
        """
        # perform segmentations for each year in each ROI
        for roi_data in preprocessed_data.values():

            for key in roi_data.keys():
                if key == "roi_path":
                    continue
                logger.info(f"roi_data[{key}]: {roi_data[key]}")
                sample_direc = roi_data[key]["sample_direc"]
                use_tta = roi_data[key]["tta"]
                use_otsu = roi_data[key]["otsu"]
                files_to_segment = self.get_files_for_seg(sample_direc)
                logger.info(f"files_to_segment: {files_to_segment}")
                if self.model_types[0] != "segformer":
                    ### mixed precision
                    from tensorflow.keras import mixed_precision
                    mixed_precision.set_global_policy("mixed_float16")
                # run model for each file
                for file_to_seg in tqdm.auto.tqdm(files_to_segment):
                    do_seg(
                        file_to_seg,
                        self.model_list,
                        self.metadata_dict,
                        self.model_types[0],
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

    def postprocess_data(self, preprocessed_data: dict, session: sessions.Session):
        if preprocessed_data == {}:
            raise Exception("No data to postprocess. You may not have selected a valid directory.")
        # save preprocessed data to session directory
        preprocessed_data_path = os.path.join(session.path, "preprocessed_data.json")
        common.write_to_json(preprocessed_data_path, preprocessed_data)
        # save segmentations & config files to session directory
        for roi_id, roi_data in preprocessed_data.items():
            roi_session_directory = common.create_directory(session.path, roi_id)
            # roi_directory = roi_data["roi_path"]
            roi_directory = preprocessed_data[roi_id]["roi_path"]
            # copy config files to session directory
            self.copy_configs(roi_directory, roi_session_directory)

            for year_key in roi_data.keys():
                print(f"Saving segmentations for year {year_key}")
                if year_key == "roi_path":
                    continue
                session.add_years(year_key)
                year_session_directory = common.create_directory(roi_session_directory, year_key)
                model_settings_path = os.path.join(year_session_directory, "model_settings.json")
                common.write_to_json(model_settings_path, preprocessed_data[roi_id][year_key])
                tiles_path = preprocessed_data[roi_id][year_key]["sample_direc"]
                session_tiles_path = common.create_directory(year_session_directory, "tiles")
                # create greyscale tif at the session_tiles_path
                greyscale_tif = make_greyscale_tif(tiles_path, year_session_directory)
                if not greyscale_tif:
                    logger.info(f"Year {year_key} could not generate a greyscale tif")
                    continue
                # delete the jpgs and xmls in tiles directory, but not the out subdirectory
                common.delete_files(r".+\.jpg$",tiles_path,recursive=False)
                common.delete_files(r".+\.xml$",tiles_path,recursive=False)
                # move the segmentations from the tiles directory to the session_tiles_path
                common.move_files_resurcively(src=tiles_path, dest=session_tiles_path)
                # delete the tiles directory
                if os.path.basename(tiles_path).lower() == "tiles":
                    os.rmdir(tiles_path)

                class_mapping = map_functions.get_class_mapping(session.classes)
                class_masks_filenames = map_functions.get_existing_class_files(year_session_directory, session.classes)
                if not class_masks_filenames:
                    class_masks_filenames = map_functions.generate_class_masks(greyscale_tif, class_mapping, year_session_directory)

        print(f"Segmentations saved at {session.path}")
        session.save(session.path)

    def copy_configs(self, src: str, dst: str) -> None:
        """
        Copies 'config.geojson' and 'config.json' files from the source to the destination directories.

        Args:
            src (str): The path to the directory containing the source files.
            dst (str): The path to the directory where the files will be copied.

        Returns:
            None.

        Raises:
            None.
        """
        # copy config.geojson and config.json files from souce to destination directories
        config_gdf_path = common.find_config_json(src, r"config_gdf.*\.geojson")
        config_json_path = common.find_config_json(src, r"^config\.json$")
        dst_file = os.path.join(dst, "config_gdf.geojson")
        logger.info(f"dst_config_gdf: {dst_file}")
        shutil.copy(config_gdf_path, dst_file)
        dst_file = os.path.join(dst, "config.json")
        logger.info(f"dst_config.json: {dst_file}")
        shutil.copy(config_json_path, dst_file)

    def get_classes(self, model_directory_path: str) -> list:
        """
        Reads the 'classes.txt' file from the given model directory path and returns a list of classes.

        Args:
            model_directory_path (str): The path to the directory containing the model files.

        Returns:
            list: A list of classes.

        Raises:
            None.
        """
        class_path = os.path.join(model_directory_path, "classes.txt")
        classes = common.read_text_file(class_path)
        return classes

    def get_downloaded_models_dir(self) -> str:
        """
        Returns the full path to the downloaded models directory, creating it if necessary.

        Returns:
            str: full path to downloaded_models directory
        """

        # directory to hold downloaded models from Zenodo
        script_dir = os.path.dirname(os.path.abspath(__file__))
        downloaded_models_path = os.path.join(script_dir, "downloaded_models")
        os.makedirs(downloaded_models_path, exist_ok=True)
        return downloaded_models_path

    def get_weights_list(self, model_path: str, model_choice: str = "BEST"):
        """
        Returns a list of the weights files (.h5) within the weights directory, based on the model choice specified.

        Args:
            model_path (str): The path to the directory containing the model
            model_choice (str): The type of model weights to return. Possible choices are "BEST" (default) or "ENSEMBLE".

        Returns:
            A list of weights files (.h5) within the weights directory, based on the model choice specified.
        """
        if model_choice == "ENSEMBLE":
            weights_list = glob(model_path + os.sep + "*.h5")
        elif model_choice == "BEST":
            # read model name (fullmodel.h5) from BEST_MODEL.txt
            with open(model_path + os.sep + "BEST_MODEL.txt") as f:
                model_name = f.readlines()
            weights_list = [model_path + os.sep + model_name[0]]
        logger.info(f"{model_choice}: {len(weights_list)} weights_list: {weights_list}")
        return weights_list

    def prepare_model(self, model_implementation: str, model_id: str):
        """
        Prepares the model for use by downloading the required files and loading the model.

        Args:
            model_implementation (str): The model implementation name.
            model_id (str): The ID of the model.
        """
        # create the model directory
        self.weights_directory = self.get_model_directory(model_id)
        logger.info(f"self.weights_directory:{self.weights_directory}")

        self.download_model(model_implementation, model_id, self.weights_directory)
        weights_list = self.get_weights_list(
            self.weights_directory, model_implementation
        )

        # Load the model from the config files
        model, model_list, config_files, model_types = self.get_model(weights_list)

        self.model_types = model_types
        self.model_list = model_list
        self.metadata_dict = self.get_metadatadict(
            weights_list, config_files, model_types
        )
        logger.info(f"self.metadatadict: {self.metadata_dict}")

    def get_metadatadict(
        self, weights_list: list, config_files: list, model_types: list
    ) -> dict:
        """Returns a dictionary containing metadata about the models.

        Args:
            weights_list (list): A list of model weights.
            config_files (list): A list of model configuration files.
            model_types (list): A list of model types.

        Returns:
            dict: A dictionary containing metadata about the models. The keys
            are 'model_weights', 'config_files', and 'model_types', and the
            values are the corresponding input lists.

        Example:
            weights = ['weights1.h5', 'weights2.h5']
            configs = ['config1.json', 'config2.json']
            types = ['unet', 'resunet']
            metadata = get_metadatadict(weights, configs, types)
            print(metadata)
            # Output: {'model_weights': ['weights1.h5', 'weights2.h5'],
            #          'config_files': ['config1.json', 'config2.json'],
            #          'model_types': ['unet', 'resunet']}
        """
        metadatadict = {}
        metadatadict["model_weights"] = weights_list
        metadatadict["config_files"] = config_files
        metadatadict["model_types"] = model_types
        return metadatadict

    def get_model_dict(self):
        return self.model_dict

    def run_model(
        self,
        model_implementation: str,
        session_name: str,
        src_directory: str,
        model_id: str,
        use_GPU: str,
        use_otsu: bool,
        use_tta: bool,
    ):

        logger.info(f"ROI directory: {src_directory}")
        logger.info(f"session name: {session_name}")
        print(f"Running model {model_id}")
        self.prepare_model(model_implementation, model_id)
        classes = self.get_classes(self.weights_directory)
        model_dict = {
            "sample_direc": None,
            "use_GPU": use_GPU,
            "implementation": model_implementation,
            "model_type": model_id,
            "otsu": use_otsu,
            "tta": use_tta,
            "classes": classes,
        }
        logger.info(f"model_dict: {model_dict}")
        # create a session
        session = sessions.Session()
        sessions_path = common.create_directory(os.getcwd(), "sessions")
        session_path = common.create_directory(sessions_path, session_name)

        session.classes = classes
        session.path = session_path
        session.name = session_name
        print(f"Preprocessing the data at {src_directory}")
        preprocessed_data = self.preprocess_data(src_directory, model_dict, session)
        if preprocessed_data == {}:
            logger.error(f"No data to process at {src_directory}")
            raise Exception(f"No data to process at {src_directory}")

        self.compute_segmentation(preprocessed_data)
        self.postprocess_data(preprocessed_data, session)
        # save session data after postprocessing data
        session.save(session_path)
        print(f"Session saved at {session_path}")
