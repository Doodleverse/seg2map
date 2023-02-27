import os
import re
from glob import glob
from typing import List

from osgeo import gdal


def delete_files(pattern, path):
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
        >>> delete_files(r'\.txt$', '/path/to/directory')
        ['/path/to/directory/file1.txt', '/path/to/directory/subdir/file2.txt']
    """
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")

    deleted_files = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if re.match(pattern, filename):
                full_path = os.path.join(dirpath, filename)
                os.remove(full_path)
                deleted_files.append(full_path)

    return deleted_files

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

def get_merged_multispectral(src_path: str,group_size:int=10) -> str:
    """
    Merges multiple GeoTIFF files into a single VRT file and returns the path of the merged file.

    This function looks for all GeoTIFF files in the specified directory, except for any files that contain the string "merged_multispectral" in their name. It groups the remaining files into batches of 4, and merges each batch into a separate VRT file. Finally, it merges all the VRT files into a single VRT file and returns the path of the merged file.

    Parameters:
    - src_path (str): The path of the directory containing the GeoTIFF files.
    - group_size (int): The size of the group of files to merge together

    Returns:
    - The path of the merged VRT file.
    """
    tif_files = glob(os.path.join(src_path, "*.tif"))
    tif_files = [file for file in tif_files if "merged_multispectral" not in file]
    print(f"Found {len(tif_files)} GeoTIFF files in {src_path}")
    print(f"tif_files: {tif_files}")
    merged_files = []
    for idx, files in enumerate(group_files(tif_files, group_size)):
        filename = f"merged_multispectral_{idx}.vrt"
        dst_path = os.path.join(src_path, filename)
        merged_tif = merge_files(files, dst_path, create_jpg=False)
        merged_files.append(merged_tif)

    print(f"merged_files {merged_files}")
    dst_path = os.path.join(src_path, " merged_multispectral.vrt")
    merged_file = merge_files(merged_files, dst_path)
    # delete intermediate merged tifs and vrts
    pattern = ".*merged_multispectral_\d+.*"
    deleted_files = delete_files(pattern, src_path)
    print(f"deleted_files {deleted_files}")
    return merged_file

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
            resampleAlg="average", srcNodata=0, VRTNodata=0
        )
        # creates a virtual world file using all the tifs and overwrites any pre-existing .vrt
        virtual_dataset = gdal.BuildVRT(dest_path, src_files, options=vrt_options)
        # flushing the cache causes the vrt file to be created
        virtual_dataset.FlushCache()
        # reset the dataset object
        virtual_dataset = None

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


if __name__ == "__main__":
    # set this to the file full of tif files 
    # NOTE: each tif file must contain a corresponding aux.xml file
    dir_path = r"C:\1_USGS\4_seg2map\seg2map\data\demo\ID_cXfdtk_dates_2010-01-01_to_2010-12-31\multiband\2010"
    src_path = os.path.abspath(dir_path)

    # try changing the group size to see if it changes the performance
    merged_file = get_merged_multispectral(src_path,group_size=15)
    print(f"Merged multispectral File: {merged_file}")