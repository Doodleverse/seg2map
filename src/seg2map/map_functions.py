import os
from typing import Set, Union, List
from base64 import b64encode
from PIL import Image
from io import BytesIO
from base64 import encodebytes
from ipyleaflet import ImageOverlay
from PIL import Image

from seg2map import common

# convert greyscale tif to png files that can be rendered on the map
file =r'C:\1_USGS\4_seg2map\seg2map\data\fresh_downloads\ID_NVBbrh_dates_2010-01-01_to_2013-12-31\multiband\2010\Mosaic.tif'
save_path=r'C:\1_USGS\4_seg2map\seg2map'

def load_classes_on_map(file,save_path:str):
    # get bounds of tif 
    bounds = common.get_bounds(file)
    # get class names to create class mapping
    names = ['bareland', 'rangeland', 'development', 'road', 'tree', 'water', 'agricultural', 'building', 'nodata']
    class_mapping = get_class_mapping(names)
    # save mask for each class in tif as a separate png
    # i should also include where these class masks should be saved
    class_masks = save_class_masks(file,class_mapping,save_path)
    # for each class mask png create an image overlay
    # combine mask name with save path
    for filename in class_masks:
        file_path = os.path.join(save_path,filename)
        print(filename)
        print(type(filename))
        # combine mask name with save path
        image_overlay=get_overlay_for_image(file_path,bounds,filename.split(".")[0])
        # seg2map.map.add_layer(image_overlay)


def get_class_mapping(names:List[str])->dict:
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

def save_class_masks(file:str,class_mapping:dict,save_path:str):
    img_gray=Image.open(file)
    # unique colors: [(count,unique_color_value)....]
    unique_colors=img_gray.getcolors()
    # Create a new color map for the masked pixels
    color_map = {
        0: (255, 0, 0),     # red
        1: (0, 255, 0),     # green
        2: (0, 0, 255),     # blue
        3: (255, 255, 0),   # yellow
        4: (255, 0, 255),   # magenta
        5: (0, 255, 255),   # cyan
        6: (128, 0, 0),     # maroon
        7: (0, 128, 0),     # dark green
        8: (0, 0, 128),     # navy
        9: (128, 128, 128), # gray
        10: (255, 255, 255) # white
    }
    # for each unique color in file create a mask with the rest of the pixels being transparent
    files_saved=[]
    for i, (count, color) in enumerate(unique_colors):
        filename = class_mapping[color]
        image_name=f"{filename}.png"
        mask = img_gray.point(lambda x: 255 * (x == color))
        mask_img = Image.new("RGBA", (img_gray.width, img_gray.height), (0, 0, 0, 0))
        mask_img.putdata([(color_map[i] + (255,) if pixel == 255 else (0, 0, 0, 0)) for pixel in mask.getdata()])
        # Save the mask image to disk with a unique filename
        img_path = os.path.join(save_path,image_name)
        mask_img.save(img_path)
        files_saved.append(image_name)
    return files_saved

def get_uri(data: bytes,scheme: str = "image/png") -> str:
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

def get_overlay_for_image(image,bounds,name:str):
    # use pillow to open the image
    img_data = Image.open(image)
    # convert image to bytes
    img_bytes=convert_image_to_bytes(img_data,file_format='png')
    # create a uri from bytes
    uri = get_uri(img_bytes,"image/png")
    # create image overlay from uri
    return ImageOverlay(url=uri, bounds=bounds, name=name)

def convert_image_to_bytes(image,file_format:str='png'):
    file_format = "PNG" if file_format.lower() == 'png' else "JPEG"
    print(file_format)
    f = BytesIO()
    image.save(f, file_format)

    # get the bytes from the bytesIO object
    return f.getvalue()
