# standard imports
import os, json
import asyncio
import platform

# external imports
from glob import glob

# import tqdm
from tqdm.auto import tqdm as auto_tqdm
import tqdm.asyncio
import zipfile
import requests
import aiohttp

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from doodleverse_utils.prediction_imports import do_seg
from doodleverse_utils.model_imports import dice_coef_loss

# Import the architectures for following models from doodleverse_utils
from doodleverse_utils.model_imports import (
    simple_resunet,
    custom_resunet,
    custom_unet,
    simple_unet,
    simple_resunet,
    simple_satunet,
    segformer,
)
from joblib import Parallel, delayed


def get_model_dir(parent_directory: str, dir_name: str) -> str:
    """returns full path to directory named dir_name and if it doesn't exist
    creates new directory dir_name within parent directory

    Args:
        parent_directory (str): directory to create new directory dir_name within
        dir_name (str): name of directory to get full path to

    Returns:
        str: full path to dir_name directory
    """
    new_dir = os.path.join(parent_directory, dir_name)
    if not os.path.isdir(new_dir):
        print(f"Creating {new_dir}")
        os.mkdir(new_dir)
    return new_dir


def request_available_files(zenodo_id: str) -> list:
    """returns list of available downloadable files for zenodo_id

    Args:
        zenodo_id (str): id of zenodo release

    Returns:
        list: list of available files downloadable for zenodo_id
    """
    # Send request to zenodo for selected model by zenodo_id
    root_url = "https://zenodo.org/api/records/" + zenodo_id
    r = requests.get(root_url)
    # get list of all files associated with zenodo id
    js = json.loads(r.text)
    files = js["files"]
    return files


def is_zipped_release(files: list) -> bool:
    """returns True if zenodo id contains 'rgb.zip' file otherwise returns false

    Args:
        files (list): list of available files for download

    Returns:
        bool: returns True if zenodo id contains 'rgb.zip' file otherwise returns false
    """
    zipped_model_list = [f for f in files if f["key"].endswith("rgb.zip")]
    if zipped_model_list == []:
        return False
    return True


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, "wb") as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def get_url_dict_to_download(models_json_dict: dict) -> dict:
    """Returns dictionary which contains
    paths to save downloaded files to matched with urls to download files

    each key in returned dictionary contains a full path to a file
    each value in returned dictionary contains url to download file
    ex.
    {'C:\Home\Project\file.json':"https://website/file.json"}

    Args:
        models_json_dict (dict): full path to files and links

    Returns:
        dict: full path to files and links
    """
    url_dict = {}
    for save_path, link in models_json_dict.items():
        if not os.path.isfile(save_path):
            url_dict[save_path] = link
        json_filepath = save_path.replace("_fullmodel.h5", ".json")
        if not os.path.isfile(json_filepath):
            json_link = link.replace("_fullmodel.h5", ".json")
            url_dict[json_filepath] = json_link

    return url_dict


async def fetch(
    session: aiohttp.client.ClientSession, url: str, save_path: str
) -> None:
    """downloads the file at url to be saved at save_path. Generates tqdm progress bar
    to track download progress of file

    Args:
        session (aiohttp.client.ClientSession): session with server that files are downloaded from
        url (str): url to file to download
        save_path (str): full path where file will be saved
    """
    model_name = url.split("/")[-1]
    chunk_size: int = 2048
    async with session.get(url, timeout=600) as r:
        content_length = r.headers.get("Content-Length")
        if content_length is not None:
            content_length = int(content_length)
            with open(save_path, "wb") as fd:
                with auto_tqdm(
                    total=content_length,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {model_name}",
                    initial=0,
                    ascii=False,
                ) as pbar:
                    async for chunk in r.content.iter_chunked(chunk_size):
                        fd.write(chunk)
                        pbar.update(len(chunk))
        else:
            with open(save_path, "wb") as fd:
                async for chunk in r.content.iter_chunked(chunk_size):
                    fd.write(chunk)


async def fetch_all(session: aiohttp.client.ClientSession, url_dict: dict) -> None:
    """concurrently downloads all urls in url_dict within a single provided session

    Args:
        session (aiohttp.client.ClientSession): session with server that files are downloaded from
        url_dict (dict): dictionary with keys as full path to file to download and value being ulr to
        file to download
    """
    tasks = []
    for save_path, url in url_dict.items():
        task = asyncio.create_task(fetch(session, url, save_path))
        tasks.append(task)
    await tqdm.asyncio.asyncio.gather(*tasks)


async def async_download_urls(url_dict: dict) -> None:
    # error raised if downloads dont's complete in 600 seconds (10 mins)
    async with aiohttp.ClientSession(raise_for_status=True, timeout=600) as session:
        await fetch_all(session, url_dict)


def run_async_download(url_dict: dict) -> None:
    """concurrently downloads all thr urls in url_dict

    Args:
        url_dict (dict): dictionary with keys as full path to file to download and value being ulr to
        file to download
    """
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # wait for async downloads to complete
    asyncio.run(async_download_urls(url_dict))


def download_zip(url: str, save_path: str, chunk_size: int = 128) -> None:
    """Downloads the zipped model from the given url to the save_path location.
    Args:
        url (str): url to zip directory  to download
        save_path (str): directory to save model
        chunk_size (int, optional):  Defaults to 128.
    """
    # make an HTTP request within a context manager
    with requests.get(url, stream=True) as r:
        # check header to get content length, in bytes
        total_length = int(r.headers.get("Content-Length"))
        with open(save_path, "wb") as fd:
            with auto_tqdm(
                total=total_length,
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


def download_zipped_model(model_direc: str, url: str) -> str:
    """download a zipped model from zenodo located at url

    'rgb.zip' is name of directory containing model online
    this function is used to download older styles of zenodo releases

    Args:
        model_direc (str): full path to directory to save model to
        url (str): url to download model from

    Returns:
        str: full path to unzipped model directory
    """
    # 'rgb.zip' is name of directory containing model online
    filename = "rgb"
    # outfile: full path to directory containing model files
    # example: 'c:/model_name/rgb'
    outfile = model_direc + os.sep + filename
    if os.path.exists(outfile):
        print(f"\n Found model weights directory: {os.path.abspath(outfile)}")
    # if model directory does not exist download zipped model from Zenodo
    if not os.path.exists(outfile):
        print(f"\n Downloading to model weights directory: {os.path.abspath(outfile)}")
        zip_file = filename + ".zip"
        zip_folder = model_direc + os.sep + zip_file
        print(f"Retrieving model {url} ...")
        download_zip(url, zip_folder)
        print(f"Unzipping model to {model_direc} ...")
        with zipfile.ZipFile(zip_folder, "r") as zip_ref:
            zip_ref.extractall(model_direc)
        print(f"Removing {zip_folder}")
        os.remove(zip_folder)

    # set weights dir to sub directory (rgb) containing model files
    model_direc = os.path.join(model_direc, "rgb")

    # Ensure all files are unzipped
    with os.scandir(model_direc) as it:
        for entry in it:
            if entry.name.endswith(".zip"):
                with zipfile.ZipFile(entry, "r") as zip_ref:
                    zip_ref.extractall(model_direc)
                os.remove(entry)
    return model_direc


def download_BEST_model(files: list, model_direc: str) -> None:
    """downloads best model from zenodo.

    Args:
        files (list): list of available files for zenodo release
        model_direc (str): directory of model to download

    Raises:
        FileNotFoundError: if model filename in 'BEST_MODEL.txt' does not
        exist online
    """
    # dictionary to hold urls and full path to save models at
    models_json_dict = {}
    # retrieve best model text file
    best_model_json = [f for f in files if f["key"].strip() == "BEST_MODEL.txt"][0]
    best_model_txt_path = os.path.join(model_direc, "BEST_MODEL.txt")
    # if BEST_MODEL.txt file not exist download it
    if not os.path.isfile(best_model_txt_path):
        download_url(
            best_model_json["links"]["self"],
            best_model_txt_path,
        )

    # read in BEST_MODEL.txt file
    with open(best_model_txt_path) as f:
        best_model_filename = f.read().strip()

    print(f"Best Model filename: {best_model_filename}")
    # check if json and h5 file in BEST_MODEL.txt exist
    model_json = [f for f in files if f["key"].strip() == best_model_filename]
    if model_json == []:
        FILE_NOT_ONLINE_ERROR = (
            f"File {best_model_filename} not found online. Raise an issue on Github"
        )
        raise FileNotFoundError(FILE_NOT_ONLINE_ERROR)
    # path to save model
    outfile = os.path.join(model_direc, best_model_filename)
    # path to save file and json data associated with file saved to dict
    models_json_dict[outfile] = model_json[0]["links"]["self"]
    url_dict = get_url_dict_to_download(models_json_dict)
    # if any files are not found locally download them asynchronous
    if url_dict != {}:
        run_async_download(url_dict)


def download_ENSEMBLE_model(files: list, model_direc: str) -> None:
    """downloads all models from zenodo.

    Args:
        files (list): list of available files for zenodo release
        model_direc (str): directory of model to download
    """
    # dictionary to hold urls and full path to save models at
    models_json_dict = {}
    # list of all models
    all_models = [f for f in files if f["key"].endswith(".h5")]
    # check if all h5 files in files are in model_direc
    for model_json in all_models:
        outfile = model_direc + os.sep + model_json["links"]["self"].split("/")[-1]
        # path to save file and json data associated with file saved to dict
        models_json_dict[outfile] = model_json["links"]["self"]
    url_dict = get_url_dict_to_download(models_json_dict)
    # if any files are not found locally download them asynchronous
    if url_dict != {}:
        run_async_download(url_dict)


def get_weights_list(model_choice: str, weights_direc: str) -> list:
    """Returns of list of full paths to weights files(.h5) within weights_direc

    Args:
        model_choice (str): 'ENSEMBLE' or 'BEST'
        weights_direc (str): full path to directory containing model weights

    Returns:
        list: list of full paths to weights files(.h5) within weights_direc
    """
    if model_choice == "ENSEMBLE":
        return glob(weights_direc + os.sep + "*.h5")
    elif model_choice == "BEST":
        with open(weights_direc + os.sep + "BEST_MODEL.txt") as f:
            w = f.readlines()
        return [weights_direc + os.sep + w[0]]


def get_metadatadict(weights_list: list, config_files: list, model_names: list) -> dict:
    """returns dictionary of model weights,config_files, and model_names

    Args:
        weights_list (list): list of full paths to weights files(.h5)
        config_files (list): list of full paths to config files(.json)
        model_names (list): list of model names

    Returns:
        dict: dictionary of model weights,config_files, and model_names
    """
    metadatadict = {}
    metadatadict["model_weights"] = weights_list
    metadatadict["config_files"] = config_files
    metadatadict["model_names"] = model_names
    return metadatadict


def get_config(weights_list: list) -> dict:
    """loads contents of config json files
    that have same name of h5 files in weights_list

    Args:
        weights_list (list): weight files(.h5) in weights_list

    Returns:
        dict: contents of config json files that have same name of h5 files in weights_list
    """
    weights_file = weights_list[0]
    configfile = (
        weights_file.replace(".h5", ".json").replace("weights", "config").strip()
    )
    if "fullmodel" in configfile:
        configfile = configfile.replace("_fullmodel", "").strip()
    with open(configfile.strip()) as f:
        config = json.load(f)
    return config


def get_model(weights_list: list):
    """Loads models in from weights list and loads in corresponding config file
    for each model weights file(.h5) in weights_list

    Args:
        weights_list (list): full path to model weights files(.h5)

    Raises:
        Exception: raised if weights_list is empty
        Exception: An unknown model type was loaded from any of weights files in
        weights_list

    Returns:
       model, model_list, config_files, model_names
    """
    model_list = []
    config_files = []
    model_names = []
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
        TARGET_SIZE = config.get("TARGET_SIZE")
        MODEL = config.get("MODEL")
        NCLASSES = config.get("NCLASSES")
        KERNEL = config.get("KERNEL")
        STRIDE = config.get("STRIDE")
        FILTERS = config.get("FILTERS")
        N_DATA_BANDS = config.get("N_DATA_BANDS")
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
            # Get the selected model based on the weights file's MODEL key provided
            # create the model with the data loaded in from the weights file
            # Load in the model from the weights which is the location of the weights file
            model = tf.keras.models.load_model(weights)
        except BaseException:
            if MODEL == "resunet":
                model = custom_resunet(
                    (TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    FILTERS,
                    nclasses=NCLASSES,
                    kernel_size=(KERNEL, KERNEL),
                    strides=STRIDE,
                    dropout=DROPOUT,  # 0.1,
                    dropout_change_per_layer=DROPOUT_CHANGE_PER_LAYER,  # 0.0,
                    dropout_type=DROPOUT_TYPE,  # "standard",
                    use_dropout_on_upsampling=USE_DROPOUT_ON_UPSAMPLING,  # False,
                )
            elif MODEL == "unet":
                model = custom_unet(
                    (TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    FILTERS,
                    nclasses=NCLASSES,
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
                    (TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel=(2, 2),
                    nclasses=NCLASSES,
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
                    (TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel=(2, 2),
                    nclasses=NCLASSES,
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
                    (TARGET_SIZE[0], TARGET_SIZE[1], N_DATA_BANDS),
                    kernel=(2, 2),
                    num_classes=NCLASSES,  # [NCLASSES+1 if NCLASSES==1 else NCLASSES][0],
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
                for k in range(NCLASSES):
                    id2label[k] = str(k)
                model = segformer(id2label, num_classes=NCLASSES)
                # model.compile(optimizer='adam')
            else:
                raise Exception(
                    f"An unknown model type {MODEL} was received. Please select a valid model."
                )
            # Load in custom loss function from doodleverse_utils
            # Load metrics mean_iou, dice_coef from doodleverse_utils
            # if MODEL!='segformer':
            #     model.compile(
            #         optimizer="adam", loss=dice_coef_loss(NCLASSES)
            #     )  # , metrics = [iou_multi(NCLASSES), dice_multi(NCLASSES)])
            weights = weights.strip()
            model.load_weights(weights)

        model_names.append(MODEL)
        model_list.append(model)
        config_files.append(configfile)
    return model, model_list, config_files, model_names


def sort_files(sample_direc: str) -> list:
    """returns list of sorted filenames in sample_direc

    Args:
        sample_direc (str): full path to directory of imagery/npz
        to run models on

    Returns:
        list: list of sorted filenames in sample_direc
    """
    # prepares data to be predicted
    sample_filenames = sorted(glob(sample_direc + os.sep + "*.*"))

    if sample_filenames[0].split(".")[-1] == "npz":
        sample_filenames = sorted(tf.io.gfile.glob(sample_direc + os.sep + "*.npz"))
    else:
        sample_filenames = sorted(tf.io.gfile.glob(sample_direc + os.sep + "*.jpg"))
        if len(sample_filenames) == 0:
            sample_filenames = sorted(glob(sample_direc + os.sep + "*.png"))
    return sample_filenames


# =========================================================


def compute_segmentation(
    TARGET_SIZE: tuple,
    N_DATA_BANDS: int,
    NCLASSES: int,
    MODEL,
    sample_direc: str,
    model_list: list,
    metadatadict: dict,
    do_parallel: bool,
    profile: str,
) -> None:
    """applies models in model_list to directory of imagery in sample_direc.
    imagery will be resized to TARGET_SIZE and should contain number of bands specified by
    N_DATA_BANDS. The outputted segmentation will contain number of classes corresponding to NCLASSES.
    Outputted segmented images will be located in a new subdirectory named 'out' created within sample_direc.
    Args:
        TARGET_SIZE (tuple):imagery will be resized to this size
        N_DATA_BANDS (int): number of bands in imagery
        NCLASSES (int): number of classes used in segmentation model
        sample_direc (str): full path to directory containing imagery to segment
        model_list (list): list of loaded models
        metadatadict (dict): config files, model weight files, and names of each model in model_list
    """
    # look for TTA config
    if "TESTTIMEAUG" not in locals():
        TESTTIMEAUG = False
    WRITE_MODELMETADATA = False
    OTSU_THRESHOLD = False

    # Read in the image filenames as either .npz,.jpg, or .png
    files_to_segment = sort_files(sample_direc)
    sample_direc = os.path.abspath(sample_direc)
    # Compute the segmentation for each of the files

    if do_parallel:

        from joblib import Parallel, delayed

        w = Parallel(n_jobs=-1, verbose=1)(
            delayed(
                do_seg(
                    file_to_seg,
                    model_list,
                    metadatadict,
                    MODEL,
                    sample_direc,
                    NCLASSES,
                    N_DATA_BANDS,
                    TARGET_SIZE,
                    TESTTIMEAUG,
                    WRITE_MODELMETADATA,
                    OTSU_THRESHOLD,
                    profile,
                )
            )()
            for file_to_seg in files_to_segment
        )

    else:

        for file_to_seg in tqdm.auto.tqdm(files_to_segment):
            do_seg(
                file_to_seg,
                model_list,
                metadatadict,
                MODEL,
                sample_direc=sample_direc,
                NCLASSES=NCLASSES,
                N_DATA_BANDS=N_DATA_BANDS,
                TARGET_SIZE=TARGET_SIZE,
                TESTTIMEAUG=TESTTIMEAUG,
                WRITE_MODELMETADATA=WRITE_MODELMETADATA,
                OTSU_THRESHOLD=OTSU_THRESHOLD,
                profile=profile,
            )
