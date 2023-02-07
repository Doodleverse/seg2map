import os
import threading
import zipfile
import concurrent.futures


def remove_zip(path):
    # Get a list of all the zipped files in the directory
    zipped_files = [
        os.path.join(path, f) for f in os.listdir(path) if f.endswith(".zip")
    ]
    # Remove each zip file
    for zipped_file in zipped_files:
        os.remove(zipped_file)


def unzip(path):
    # Get a list of all the zipped files in the directory
    zipped_files = [
        os.path.join(path, f) for f in os.listdir(path) if f.endswith(".zip")
    ]
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


def remove_zip_files(paths):
    # Create a thread pool with a fixed number of threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit a remove_zip task for each directory
        futures = [executor.submit(remove_zip, path) for path in paths]

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)


def get_subdirs(parent_dir):
    # Get a list of all the subdirectories in the parent directory
    subdirs = [
        os.path.join(parent_dir, d)
        for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]
    return subdirs

def unzip_data(parent_dir:str):
    subdirs = get_subdirs(parent_dir)
    unzip_files(subdirs)
    # remove_zip_files(subdirs)
 

parent_dir = r"C:\1_USGS\5_Doodleverse\1_Seg2Map_fork\seg2map\ROIs\ROI9"
unzip_data(parent_dir)
# roi_8 = r"C:\1_USGS\5_Doodleverse\1_Seg2Map_fork\seg2map\ROIs\ROI8\multiband"
# unzip(roi_8)