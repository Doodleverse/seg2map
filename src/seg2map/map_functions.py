import os
import logging
from typing import Tuple, List
from PIL import Image
from io import BytesIO
import numpy as np
from base64 import encodebytes
from ipyleaflet import ImageOverlay
from PIL import Image
from time import perf_counter

from seg2map import common

logger = logging.getLogger(__name__)


def time_func(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds to run.")
        return result

    return wrapper


def get_existing_class_files(dir_path: str, class_names: list[str]) -> list[str]:
    """
    Given a directory path and a list of class names, returns a list of paths to PNG files in that directory
    whose filenames match the class names.

    Args:
        dir_path (str): The path to the directory to search for PNG files.
        class_names (list[str]): A list of class names to match against PNG filenames.

    Returns:
        list[str]: A list of paths to PNG files in the directory whose filenames match the class names.
    """
    existing_files = []
    for class_name in class_names:
        filename = f"{class_name}.png"
        file_path = os.path.join(dir_path, f"{class_name}.png")
        if os.path.isfile(file_path):
            existing_files.append(filename)
    return existing_files


def get_class_masks_overlay(
    tif_file: str, mask_output_dir: str, classes: List[str], year: str, roi_id: str
) -> List:
    """
    Given a path to a TIFF file, create binary masks for each class in the file and
    return a list of image overlay layers that can be used to display the masks over
    the original image.

    Args:
        tif_file (str): The path to the input TIFF file.
        mask_output_dir (str): The path to the directory where the output mask images
            will be saved.
        classes (List[str]): A list of class names to include in the masks.
        year(str): year that tif was created

    Returns:
        A list of image overlay layers, one for each class mask.
    """
    logger.info(f"tif_file: {tif_file}")
    # get bounds of tif
    bounds = common.get_bounds(tif_file)

    # get class names to create class mapping
    class_mapping = get_class_mapping(classes)

    # see if any class masks already exist
    class_masks_filenames = get_existing_class_files(mask_output_dir, classes)

    # generate binary masks for each class in tif as a separate PNG in mask_output_dir
    if not class_masks_filenames:
        class_masks_filenames = generate_class_masks(
            tif_file, class_mapping, mask_output_dir
        )

    # for each class mask PNG, create an image overlay
    layers = []
    for file_path in class_masks_filenames:
        file_path = os.path.join(mask_output_dir, file_path)
        layer_name = (
            roi_id + "_" + os.path.basename(file_path).split(".")[0] + "_" + year
        )
        # combine mask name with save path
        image_overlay = get_overlay_for_image(
            file_path, bounds, layer_name, file_format="png"
        )
        layers.append(image_overlay)
    return layers


def get_class_layers(tif_directory, classes, year, roi_id) -> List:
    # locate greyscale segmented tif in session directory
    greyscale_tif_path = common.find_file(
        tif_directory, "Mosaic_greyscale.tif", case_insensitive=True
    )
    if greyscale_tif_path is None:
        logger.warning(
            f"Does not exist {os.path.join(tif_directory, '*merged_multispectral.jp*g*')}"
        )
        return []
    # create layers for each class present in greyscale tiff
    class_layers = get_class_masks_overlay(
        greyscale_tif_path, tif_directory, classes, year, roi_id
    )
    return class_layers


def get_class_mapping(names: List[str]) -> dict:
    """Create a mapping of class names to integer labels.

    Given a list of class names, this function creates a dictionary that maps each
    class name to a unique integer label starting from 1.

    Parameters
    ----------
    names : list of str
        A list of class names to map to integer labels.

    Returns
    -------
    dict
        A dictionary mapping class names to integer labels.

    Ex:
      get_class_mapping(['water','sand'])
    {
        1:'water',
        2:'sand'
    }
    """
    class_mapping = {}
    for i, name in enumerate(names, start=1):
        class_mapping[i] = name
    return class_mapping


def generate_color_map(num_colors: int) -> dict:
    """
    Generate a color map for the specified number of colors.

    Args:
        num_colors (int): The number of colors needed.

    Returns:
        A dictionary containing a color map for the specified number of colors.
    """
    import colorsys

    # Generate a list of equally spaced hues
    hues = [i / num_colors for i in range(num_colors)]

    # Convert each hue to an RGB color tuple
    rgb_colors = [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hues]

    # Scale each RGB value to the range [0, 255] and add to the color map
    color_map = {}
    for i, color in enumerate(rgb_colors):
        color_map[i] = tuple(int(255 * c) for c in color)

    return color_map


# @time_func
def generate_class_masks(file: str, class_mapping: dict, save_path: str) -> List[str]:
    """
    Generate binary masks for each class in the given grayscale image, based on a color-to-class mapping.

    Args:
        file (str): The path to the grayscale input image file.
        class_mapping (dict): A dictionary that maps pixel colors to class names.
        save_path (str): The path to the directory where the generated mask images will be saved.

    Returns:
        List[str]: A list of filenames of the saved mask images.

    Raises:
        None

    Example:
        If file='path/to/image.tif', class_mapping={0: 'background', 1: 'water', 2: 'land'}, and save_path='path/to/masks',
        generate_class_masks(file, class_mapping, save_path) returns ['background.png', 'water.png', 'land.png'].

    """
    img_gray = Image.open(file)
    unique_colors = img_gray.getcolors()
    color_map = generate_color_map(len(unique_colors))

    # Convert the image to a NumPy array
    img_gray_np = np.array(img_gray)

    files_saved = []
    for i, (count, color) in enumerate(unique_colors):
        filename = class_mapping[color]
        image_name = f"{filename}.png"

        # Create a binary mask with 1 where the pixel color matches and 0 elsewhere
        mask = (img_gray_np == color).astype(np.uint8)

        # Create a new RGBA image with the same dimensions as the input image
        mask_img = np.zeros((img_gray.height, img_gray.width, 4), dtype=np.uint8)

        # Set the RGB values of the mask image to the corresponding color in the color map
        mask_img[..., :3] = np.array(color_map[i]) * mask[..., None]

        # Set the alpha channel to 255 where the mask is 1, and 0 elsewhere
        mask_img[..., 3] = mask * 255

        # Convert the NumPy array back to a PIL Image object
        mask_img_pil = Image.fromarray(mask_img)

        # Save the mask image to disk with a unique filename
        img_path = os.path.join(save_path, image_name)
        mask_img_pil.save(img_path)
        files_saved.append(image_name)

    return files_saved


def get_uri(data: bytes, scheme: str = "image/png") -> str:
    """Generates a URI (Uniform Resource Identifier) for a given data object and scheme.

    The data is first encoded as base64, and then added to the URI along with the specified scheme.

    Works for both RGB and RGBA imagery

    Scheme : string of character that specifies the purpose of the uri
    Available schemes for imagery:
    "image/jpeg"
    "image/png"

    Parameters
    ----------
    data : bytes
        The data object to be encoded and added to the URI.
    scheme : str, optional (default="image/png")
        The URI scheme to use for the generated URI. Defaults to "image/png".

    Returns
    -------
    str
        The generated URI, as a string.
    """
    return f"data:{scheme};base64,{encodebytes(data).decode('ascii')}"


def get_overlay_for_image(
    image_path: str, bounds: Tuple, name: str, file_format: str
) -> ImageOverlay:
    """Create an ImageOverlay object for an image file.

    Args:
        image_path (str): The path to the image file.
        bounds (Tuple): The bounding box for the image overlay.
        name (str): The name of the image overlay.
        file_format (str): The format of the image file, either 'png', 'jpg', or 'jpeg'.

    Returns:
        An ImageOverlay object.
    """
    if file_format.lower() not in ["png", "jpg", "jpeg"]:
        raise ValueError(
            f"{file_format} is not recognized. Allowed file formats are: png, jpg, and jpeg."
        )

    if file_format.lower() == "png":
        file_format = "png"
        scheme = "image/png"
    elif file_format.lower() == "jpg" or file_format.lower() == "jpeg":
        file_format = "jpeg"
        scheme = "image/jpeg"

    logger.info(f"image_path: {image_path}")
    logger.info(f"file_format: {file_format}")

    # use pillow to open the image
    img_data = Image.open(image_path)
    # convert image to bytes
    img_bytes = convert_image_to_bytes(img_data, file_format)
    # create a uri from bytes
    uri = get_uri(img_bytes, scheme)
    # create image overlay from uri
    return ImageOverlay(url=uri, bounds=bounds, name=name)


def convert_image_to_bytes(image, file_format: str = "png"):
    if file_format.lower() not in ["png", "jpg", "jpeg"]:
        raise ValueError(
            f"{file_format} is not recognized. Allowed file formats are: png, jpg, and jpeg."
        )
    file_format = "PNG" if file_format.lower() == "png" else "JPEG"
    f = BytesIO()
    image.save(f, file_format)
    # get the bytes from the bytesIO object
    return f.getvalue()
