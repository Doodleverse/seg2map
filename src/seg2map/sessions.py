import json
import os
from typing import List

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


class Session:
    """
    A class representing a session, which contains sets of classes, years, and ROI IDs.
    """

    def __init__(self, name: str = None, path: str = None):
        """
        Initializes a new Session object.

        Args:
            name (str): The name of the session. Default is None.
            path (str): The path to the directory where the session will be saved. Default is None.
        """
        self.name = name
        self.path = path
        self.classes = set()
        self.years = set()
        self.roi_ids = set()
        self.roi_info = {}

    def get_session_data(self) -> dict:
        session_data = {
            "name": self.name,
            "path": self.path,
            "classes": list(self.classes),
            "years": list(self.years),
            "roi_ids": list(self.roi_ids),
        }
        return session_data

    def get_roi_info(self, roi_id: str = None):
        if roi_id:
            return self.roi_info.get(roi_id, "")
        return self.roi_info

    def set_roi_info(self, new_roi_info: dict):
        return self.roi_info.update(new_roi_info)

    # def create_roi_directories()

    def add_classes(self, class_names: List[str]):
        """
        Adds one or more class names to the session.

        Args:
            class_names (str or iterable): The name(s) of the class(es) to add.
        """
        if isinstance(class_names, str):
            self.classes.add(class_names)
        else:
            self.classes.update(class_names)

    def add_years(self, years: List[str]):
        """
        Adds one or more years to the session.

        Args:
            years (int or str or iterable): The year(s) to add.
        """
        if isinstance(years, int):
            self.years.add(str(years))
        elif isinstance(years, str):
            self.years.add(years)
        else:
            self.years.update(years)

    def add_roi_ids(self, roi_ids: List[str]):
        """
        Adds one or more ROI IDs to the session.

        Args:
            roi_ids (str or iterable): The ROI ID(s) to add.
        """
        if isinstance(roi_ids, str):
            self.roi_ids.add(roi_ids)
        else:
            self.roi_ids.update(roi_ids)

    def find_session_file(self, path: str, filename: str = "session.json"):
        # if session.json is found in main directory then session path was identified
        session_path = os.path.join(path, filename)
        if os.path.isfile(session_path):
            return session_path
        else:
            parent_directory = os.path.dirname(path)
            json_path = os.path.join(parent_directory, filename)
            if os.path.isfile(json_path):
                return json_path
            else:
                raise ValueError(
                    f"File '{filename}' not found in the parent directory: {parent_directory} or path"
                )

    def load(self, path: str):
        """
        Loads a session from a directory.

        Args:
            path (str): The path to the session directory.
        """
        json_path = self.find_session_file(path, "session.json")
        with open(json_path, "r") as f:
            session_data = json.load(f)
            self.name = session_data.get("name")
            self.path = session_data.get("path")
            self.classes = set(session_data.get("classes", []))
            self.years = set(session_data.get("years", []))
            self.roi_ids = set(session_data.get("roi_ids", []))

    def save(self, path):
        """
        Saves the session to a directory.

        Args:
            path (str): The path to the directory where the session will be saved.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        session_data = {
            "name": self.name,
            "path": path,
            "classes": list(self.classes),
            "years": list(self.years),
            "roi_ids": list(self.roi_ids),
        }

        with open(os.path.join(path, "session.json"), "w") as f:
            json.dump(session_data, f, indent=4)

    def __str__(self):
        """
        Returns a string representation of the session.

        Returns:
            str: A string representation of the session.
        """
        return f"Session: {self.name}\nPath: {self.path}\nClasses: {self.classes}\nYears: {self.years}\nROI IDs: {self.roi_ids}\n"


# Example usage:

# Create a new session
# session = Session()
# session.classes = {'Math', 'English', 'Science'}
# session.years = {2020, 2021, 2022}
# session.roi_ids = {1, 2, 3}
# session.name = 'session1'
# session.path = '/path/to/sessions/session1'

# # Save the session
# session.save(session.path)

# # Load the session from disk
# session2 = Session()
# session2.load(session.path)
# print(os.path.abspath(session.path))

# # Check that the loaded session has the same values as the original session
# print(session2)
