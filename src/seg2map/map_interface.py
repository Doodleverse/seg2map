import os
import pathlib
from glob import glob
import json
import logging
from typing import List, Optional
from collections import defaultdict
from datetime import datetime
from typing import Union

from seg2map import common
from seg2map.roi import ROI
from seg2map import exceptions
from seg2map import exception_handler
from seg2map import downloads
from seg2map import map_functions
from seg2map import sessions

import geopandas as gpd
import tqdm
import tqdm.auto
from ipyleaflet import DrawControl, LayersControl, WidgetControl, GeoJSON
from leafmap import Map
from ipywidgets import Layout, HTML, Accordion
from ipywidgets import ToggleButton
from ipywidgets import HBox
from ipywidgets import VBox
from ipywidgets import HTML
import ipyleaflet


logger = logging.getLogger(__name__)


class Seg2Map:
    def __init__(self):
        # settings:  used to select data to download and preprocess settings
        self.settings = {
            "dates": "",
            "sitename": "",
        }
        # segmentation layers for each year
        self.seg_layers = []
        # years: available years to show for segmentations
        self.years = []
        # segmentation data associated with each ROI
        self.roi_segmentations = {}
        # classes : all classes available for segmentations
        self.classes = set()
        # original imagery layers for each year
        self.original_layers = []
        # year that have imagery downloaded
        self.years = []
        # selected_set set(str): ids of the selected rois
        self.selected_set = set()
        # selected_set set(str): ids of rois selected for deletion
        self.delete_set = set()
        # ROI objects on map
        self.rois = ROI()
        # selected layer name
        self.SELECTED_LAYER_NAME = "selected"
        # layer that contains ROIs that will be deleted
        self.DELETE_LAYER_NAME = "delete"
        # create map if map_settings not provided else use default settings
        self.map = self.create_map()
        # create controls and add to map
        self.draw_control = self.create_DrawControl(DrawControl())
        self.draw_control.on_draw(self.handle_draw)
        self.map.add(self.draw_control)
        layer_control = LayersControl(position="topright")
        self.map.add(layer_control)
        hover_accordion_control = self.create_accordion_widget()
        self.map.add(hover_accordion_control)
        # action box is used to perform actions on selected ROIs like deletion
        self.remove_box = HBox([])
        self.action_widget = WidgetControl(widget=self.remove_box, position="topleft")
        self.map.add(self.action_widget)
        self.warning_box = HBox([])
        self.warning_widget = WidgetControl(widget=self.warning_box, position="topleft")
        self.map.add(self.warning_widget)

    def get_original_layers(
        self, layer_name: Optional[str] = None
    ) -> Union[ipyleaflet.Layer, list[ipyleaflet.Layer], None]:
        """
        Returns the first matching original layer with the specified name, or all original layers if no name is given.

        Args:
            layer_name (str, optional): The name of the layer to look for. If not specified, returns all original layers.

        Returns:
            ipyleaflet.Layer or list[ipyleaflet.Layer] or None: The first matching layer, or all layers if no name is given.
            Returns None if no layer is found.

        Examples:
            # Get the original layer with name "My Layer"
            my_layer = get_original_layers("My Layer")

            # Get all original layers
            all_layers = get_original_layers()
        """
        matching_layers = []
        if layer_name is not None:
            for layer in self.seg_layers:
                if layer_name in layer.name:
                    matching_layers.append(layer)
            return matching_layers
        else:
            return self.original_layers

    def get_seg_layers(self, layer_name: Optional[str] = None) -> List:
        """
        Returns the first matching segmentation layer with the specified name, or all segmentation layers if no name is given.

        Args:
            layer_name (str, optional): The name of the layer to look for. If not specified, returns all segmentation layers.

        Returns:
            ipyleaflet.Layer or list[ipyleaflet.Layer]: The first matching layer, or all layers if no name is given. Returns None if no layer is found.

        Examples:
            # Get the segmentation layer with name "My Layer"
            my_layer = get_seg_layer("My Layer")

            # Get all segmentation layers
            all_layers = get_seg_layer()
        """
        matching_layers = []
        if layer_name is not None:
            for layer in self.seg_layers:
                if layer_name in layer.name:
                    matching_layers.append(layer)
            return matching_layers
        else:
            return self.seg_layers

    def get_roi_segmentations(self):
        return self.roi_segmentations.copy()

    def set_roi_segmentations(
        self, roi_id: str, years: List[str], classes: List[str]
    ) -> None:
        """Sets the segmentation information for a given ROI.

        Args:
            roi_id (str): The ID of the ROI.
            years (List[str]): The years for which segmentation data is available.
            classes (List[str]): The classes for which segmentation data is available.
        """
        classes = list(classes)
        self.roi_segmentations[roi_id] = {"years": years, "classes": classes}

    def get_classes(self, roi_id=None):
        if roi_id:
            classes = self.get_roi_segmentations()[roi_id]["classes"]
            classes = set(classes)
            return classes
        return self.classes

    def get_years(self, roi_id=None):
        if roi_id:
            return self.get_roi_segmentations()[roi_id]["years"]
        return self.years

    def get_setttings(self) -> dict:
        logger.info(f"settings: {self.settings}")

        if self.settings["sitename"] == "":
            raise Exception("No sitename given")

        dates = self.settings["dates"]
        if dates == "":
            raise Exception("No dates given")
        elif dates != "":
            dates = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
            if dates[1] <= dates[0]:
                raise Exception("Dates are not correct chronological order")

        return self.settings

    def load_session_layers(self, session_path: str, classes: List[str], roi_id: str):
        # load model_settings and read roi directory from it
        model_settings_path = os.path.join(session_path, "model_settings.json")
        model_settings = common.read_json_file(model_settings_path)
        roi_path = model_settings.get("sample_direc", "")
        if not roi_path:
            raise ValueError(
                f"model_settings.json is missing sample_direc. {model_settings_path}"
            )

        roi_directory = pathlib.Path(model_settings["sample_direc"])
        # if the directory name is 'tiles' chdir to the parent directory
        if "tile" in os.path.basename(roi_directory):
            roi_directory = os.path.dirname(roi_directory)
        # Load original imagery on map
        original_jpg_path = os.path.join(roi_directory, "merged_multispectral.jpg")
        original_tif_path = os.path.join(roi_directory, "merged_multispectral.tif")
        if not os.path.isfile(original_jpg_path):
            logger.warning(f"Does not exist {original_jpg_path}")
            return
        if not os.path.isfile(original_tif_path):
            logger.warning(f"Does not exist {original_tif_path}")
            return
        logger.info(f"original_jpg_path: {original_jpg_path}")
        logger.info(f"original_tif_path: {original_tif_path}")
        # create layer name in the form of {image_name}{year}
        year_name = os.path.basename(session_path)
        layer_name = f"{roi_id}_{year_name}"
        new_layer = common.get_image_overlay(
            str(original_tif_path),
            str(original_jpg_path),
            layer_name,
            convert_RGB=True,
            file_format="jpeg",
        )
        if new_layer:
            self.original_layers.append(new_layer)
        # create layers for each class present in greyscale tiff
        class_layers = map_functions.get_class_layers(
            session_path, classes, year_name, roi_id
        )
        if class_layers:
            self.seg_layers.extend(class_layers)

    def load_session(self, session_path: str) -> None:
        """
        Loads a session onto the map, containing the segmented imagery and a model settings.json file that contains a reference to the original imagery.
        """
        self.map.default_style = {"cursor": "wait"}
        session_path = os.path.abspath(session_path)

        session = sessions.Session()
        session.load(session_path)
        classes = list(session.classes)
        years = list(session.years)
        roi_ids = list(session.roi_ids)
        if not years:
            raise Exception(f"No year directories found in session {session_path}")
        if not classes:
            raise ValueError(f"Session is missing classes {session_path}")
        if not roi_ids:
            raise ValueError(f"No ROI directories found in session. {session_path}")

        self.classes = sorted(classes)
        self.years = sorted(years)

        # if selected path is not the session name then an ROI directory was selected
        logger.info(f"session_path: {session_path}")
        logger.info(f"os.path.basename(session_path): {os.path.basename(session_path)}")
        logger.info(f"session.name: {session.name}")

        if session.name != os.path.basename(session_path):
            roi_id = os.path.basename(session_path)
            self.set_roi_segmentations(roi_id, years, classes)
            top_level_directories = [
                os.path.join(session_path, d)
                for d in os.listdir(session_path)
                if os.path.isdir(os.path.join(session_path, d))
            ]
            logger.info(f"top_level_directories: {top_level_directories}")
            for session_year_path in top_level_directories:
                self.load_session_layers(session_year_path, classes, roi_id)

        else:
            for roi_id in roi_ids:
                self.set_roi_segmentations(roi_id, years, classes)
                session_roi_directory = os.path.join(session_path, roi_id)
                top_level_directories = [
                    os.path.join(session_roi_directory, d)
                    for d in os.listdir(session_roi_directory)
                    if os.path.isdir(os.path.join(session_roi_directory, d))
                ]
                logger.info(f"top_level_directories: {top_level_directories}")
                for session_year_path in top_level_directories:
                    self.load_session_layers(session_year_path, classes, roi_id)

        # get lists of original and segmentation layers for imagery
        original_layers = self.get_original_layers()
        seg_layers = self.get_seg_layers()

        # By default load first available year of segmentations on the map
        if len(original_layers) == 0:
            logger.error(f"No imagery to load on map")
            raise Exception(f"No imagery to load on map")
        if len(seg_layers) == 0:
            logger.warning(f"No segmentations to load on map")
        # Display first available year by default
        year = original_layers[0].name.split("_")[-1]
        layers = original_layers + seg_layers
        self.load_layers_by_year(layers, year)
        self.map.default_style = {"cursor": "default"}

    def load_layers_by_year(self, layers: List[ipyleaflet.Layer], year: str) -> None:
        # load layers on map if year is in layer name
        # remove layers where year is not in layer name
        # make sure layer name does not contain 'ROI'
        year = str(year)
        if len(layers) == 0:
            logger.warning(f"No imagery to load on map for year {year}:{layers}")
        # load layers on map by year
        for layer in layers:
            # load the layer on the map if layer name contains year and layer not on map
            if year in layer.name and self.map.find_layer(layer.name) is None:
                self.map.add_layer(layer)
            if year not in layer.name and layer.name != "ROI":
                if layer in self.map.layers:
                    self.map.remove_layer(layer)

    def modify_layers_opacity_by_year(
        self, layers: list, year: str = "", opacity: float = 1.0
    ) -> None:
        """
        Modifies the opacity of all layers in the given list whose name contains the specified year.

        Args:
            layers (list): A list of layers to modify.
            year (str, optional): The year to match against layer names. Default is an empty string, which matches all layers.
            opacity (float, optional): The opacity value to set for the matching layers. Default is 1.0 (fully opaque).

        Returns:
            None: This function doesn't return anything.

        Examples:
            # Set opacity to 0.5 for all layers with "2020" in their name
            modify_layers_opacity_by_year(layers, "2020", 0.5)
        """
        year = str(year)
        # for each layer with year in its name modify the opacity
        for layer in layers:
            # load the layer on the map if layer name contains year and layer not on map
            if year in layer.name:
                self.modify_layer_opacity(layer, opacity)

    def modify_layer_opacity(
        self, layer: ipyleaflet.Layer, opacity: float = 1.0
    ) -> None:
        """
        Modifies the opacity of the specified layer on a map.

        Args:
            layer (ipyleaflet.Layer): The layer to modify.
            opacity (float, optional): The opacity value to set for the layer, between 0.0 (fully transparent) and 1.0 (fully opaque).
                Default is 1.0.

        Returns:
            None: This function doesn't return anything.

        Raises:
            TypeError: If the `layer` argument is not an instance of `ipyleaflet.Layer`.
            ValueError: If the layer is not found on the map.

        Examples:
            # Set opacity to 0.5 for the layer with name "My Layer"
            my_layer = ipyleaflet.TileLayer(name="My Layer")
            modify_layer_opacity(my_layer, 0.5)
        """
        if isinstance(layer, ipyleaflet.Layer):
            if self.map.find_layer(layer.name):
                self.map.layer_opacity(layer.name, opacity)

    def download_imagery(
        self,
    ) -> None:
        """Download imagery for the selected ROIs on the map.

        Raises:
            exceptions.Object_Not_Found: If no ROIs exist.
            Exception: If the sitename in settings already exists.

        This function checks for the existence of ROIs and selected sets, gets the settings used to download the ROIs,
        creates a directory to store the downloads, downloads the selected ROIs to the directory, and deletes any empty
        directories. Finally, it saves the configuration.
        """
        # throw error if no ROIs exist
        exception_handler.check_if_None(self.rois, "ROIs")
        # if selected set is empty throw an error
        exception_handler.check_selected_set(self.selected_set)
        selected_ids = list(self.selected_set)
        logger.info(f"selected_ids: {selected_ids}")

        # get settings used to download ROIs
        settings = self.get_setttings()
        # create directory named sitename within data directory in current working directory to hold all downloads
        site_path = common.get_site_path(settings)
        logger.info(
            f"Download in process.\nsitepath: {site_path}\nselected ids:{selected_ids}"
        )
        # download all selected ROIs on map to sitename directory
        print("Download in process")

        roi_paths = []
        with common.Timer():
            for roi_id in selected_ids:
                roi_path = downloads.create_ROI_directories(
                    site_path, roi_id, settings["dates"]
                )
                roi_paths.append(roi_path)

        with common.Timer():
            ROI_tiles = downloads.run_async_function(
                downloads.get_tiles_for_ids,
                roi_paths=roi_paths,
                rois_gdf=self.rois.gdf,
                selected_ids=selected_ids,
                dates=settings["dates"],
            )

        logger.info(f"ROI_tiles: {ROI_tiles}")
        with common.Timer():
            downloads.run_async_function(downloads.download_ROIs, ROI_tiles=ROI_tiles)

        self.save_config()

        # create merged multispectural for each year in each ROI
        common.create_merged_multispectural_for_ROIs(roi_paths)

    def create_delete_box(self, title: str = None, msg: str = None):
        padding = "0px 5px 0px 5px"  # upper, right, bottom, left
        # create title
        if title is None:
            title = "Delete Selected ROIs?"
        warning_title = HTML(f"<b>⚠️<u>{title}</u></b>")
        # create msg
        if msg is None:
            msg = "Select ROIs to be deleted then click ok"
        warning_msg = HTML(
            f"______________________________________________________________________\
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

        def close_click(change):
            if change["new"]:
                warning_content.close()
                close_button.close()
                ok_button.close()
                # revert map back to normal state
                self.exit_delete_state()

        # define a ok button that deletes selected ROI on click
        ok_button = ToggleButton(
            value=False,
            description="OK",
            tooltip="Deletes ROI",
            button_style="danger",
            layout=Layout(height="28px", width="40px", padding=padding),
        )
        # handler for ok button that deletes selected ROI on click
        def ok_click(change):
            if change["new"]:
                try:
                    logger.info(f"Deleting selected ROIs {self.delete_set}")
                    self.remove_selected_rois(self.delete_set)
                    # revert map back to normal state
                    self.exit_delete_state()
                    warning_content.close()
                    close_button.close()
                    ok_button.close()
                except Exception as error:
                    # renders error message as a box on map
                    exception_handler.handle_exception(error, self.warning_box)

        ok_button.observe(ok_click, "value")
        close_button.observe(close_click, "value")
        top_row = HBox([warning_content, close_button])
        delete_box = VBox([top_row, ok_button])
        return delete_box

    def create_map(self):
        """create an interactive map object using the map_settings
        Returns:
           ipyleaflet.Map: ipyleaflet interactive Map object
        """
        map_settings = {
            "center_point": (40.8630302395, -124.166267),
            "zoom": 13,
            "draw_control": False,
            "measure_control": False,
            "fullscreen_control": False,
            "attribution_control": True,
            "Layout": Layout(width="100%", height="100px"),
        }
        return Map(
            draw_control=map_settings["draw_control"],
            measure_control=map_settings["measure_control"],
            fullscreen_control=map_settings["fullscreen_control"],
            attribution_control=map_settings["attribution_control"],
            center=map_settings["center_point"],
            zoom=map_settings["zoom"],
            layout=map_settings["Layout"],
        )

    def create_accordion_widget(self):
        """creates a accordion style widget controller to hold data of
        a feature that was last hovered over by user on map.
        Returns:
           ipyleaflet.WidgetControl: an widget control for an accordion widget
        """
        roi_html = HTML("Hover over a ROI on the map")
        roi_html.layout.margin = "0px 20px 20px 20px"
        self.accordion = Accordion(
            children=[roi_html], titles=("Features Data", "ROI Data")
        )
        self.accordion.set_title(0, "ROI Data")
        return WidgetControl(widget=self.accordion, position="topright")

    def load_configs(self, filepath: str) -> None:
        """Loads features from geojson config file onto map and loads
        config.json file into settings. The config.json should be located into
        the same directory as the config.geojson file
        Args:
            filepath (str): full path to config.geojson file
        """
        # load geodataframe from config and load features onto map
        self.load_gdf_config(filepath)

        # path to directory to search for config_gdf.json file
        search_path = os.path.dirname(os.path.realpath(filepath))
        try:
            # create path to config.json file in search_path directory
            config_path = common.find_config_json(search_path)
            logger.info(f"Loaded json config file from {config_path}")
            # load settings from config.json file
            self.load_json_config(config_path)
        except FileNotFoundError as file_error:
            print(file_error)
            print("No settings loaded in from config file")
            logger.warning(f"No settings loaded in from config file. {file_error}")

    def load_gdf_config(self, filepath: str) -> None:
        """Load features from geodataframe located in geojson file
        located at filepath onto map.

        features in config file should contain a column named "type"
        which contains the name of one of the following possible feature types
        "roi","bbox".
        Args:
            filepath (str): full path to config.geojson
        """
        print(f"Loading {filepath}")
        logger.info(f"Loading {filepath}")
        gdf = common.read_gpd_file(filepath)
        # Extract ROIs from gdf and create new dataframe
        roi_gdf = gdf[gdf["type"] == "roi"].copy()
        exception_handler.check_if_gdf_empty(
            roi_gdf,
            "ROIs from config",
            "No ROIs were present in the config file: {filepath}",
        )
        # drop all columns except id and geometry
        columns_to_drop = list(set(roi_gdf.columns) - set(["id", "geometry"]))
        logger.info(f"Dropping columns from ROI: {columns_to_drop}")
        roi_gdf.drop(columns_to_drop, axis=1, inplace=True)
        logger.info(f"roi_gdf: {roi_gdf}")

        bounds = roi_gdf.total_bounds
        self.map.zoom_to_bounds(bounds)
        # Add geodataframe from config file to ROI
        self.rois.add_geodataframe(roi_gdf)
        self.load_feature_on_map()

    def load_json_config(self, filepath: str) -> None:
        """
        Load configuration data from a JSON file and update the object's settings and ROI settings.

        Parameters:
        filepath (str): The path to the JSON file containing the configuration data.

        Raises:
        FileNotFoundError: If the file at `filepath` cannot be found.
        Exception: If `self.rois` is None.

        """
        exception_handler.check_if_None(self.rois)
        json_data = common.read_json_file(filepath)
        # replace settings with settings from config file
        self.save_settings(**json_data["settings"])

        # replace roi_settings for each ROI with contents of config.json
        new_roi_settings = {
            str(roi_id): json_data[roi_id] for roi_id in json_data["roi_ids"]
        }
        self.rois.set_settings(new_roi_settings)

    def save_config(self, filepath: str = None) -> None:
        """saves the configuration settings of the map into two files
            config.json and config.geojson
            Saves the settings used to download imagery to config.json and
            saves the ROIs loaded on the map a config.geojson
        Args:
            file_path (str, optional): path to directory to save config files. Defaults to None.
        Raises:
            Exception: raised if self.settings is missing
            Exception: raised if self.rois is missing
            Exception: raised if selected set is empty because no ROIs were selected
        """
        exception_handler.config_check_if_none(self.settings, "settings")
        exception_handler.config_check_if_none(self.rois, "ROIs")
        exception_handler.check_selected_set(self.selected_set)

        roi_settings = self.rois.get_settings()
        # if ROIs have no settings load settings into the rois
        if roi_settings == {}:
            default_file_path = filepath or os.path.join(os.getcwd(), "data")
            roi_settings = common.create_roi_settings(
                self.settings, self.selected_set, default_file_path
            )
            self.rois.set_settings(roi_settings)
            logger.info(
                f"No roi settings found. Created ROI settings: {self.rois.get_settings()}"
            )

        # create dictionary to be saved to config.json
        config_json = common.create_json_config(roi_settings, self.settings)
        logger.info(f"config_json: {config_json} ")

        # save all selected rois to config geodataframe
        selected_rois = self.get_selected_rois(config_json["roi_ids"])
        logger.info(f"selected_rois: {selected_rois} ")
        config_gdf = common.create_config_gdf(selected_rois)
        logger.info(f"config_gdf: {config_gdf} ")

        # save config files to the provided filepath
        if filepath is not None:
            # save to config.json
            common.config_to_file(config_json, filepath)
            # save to config_gdf.geojson
            common.config_to_file(config_gdf, filepath)
            print(f"Saved config files for each ROI to {filepath}")
        elif filepath is None:
            is_downloaded = common.were_rois_downloaded(
                roi_settings, config_json["roi_ids"]
            )
            logger.info(f"Rois were {is_downloaded} downloaded")
            # data has been downloaded before so inputs have keys 'filepath' and 'sitename'
            if is_downloaded == True:
                # for each ROI save two config file to the ROI's directory
                for roi_id in config_json["roi_ids"]:
                    sitename = str(config_json[roi_id]["sitename"])
                    roi_name = str(config_json[roi_id]["roi_name"])
                    ROI_path = os.path.join(
                        config_json[roi_id]["filepath"], sitename, roi_name
                    )
                    logger.info(f"ROI_path{ROI_path}")
                    # save settings to config.json
                    common.config_to_file(config_json, ROI_path)
                    # save geodataframe to config_gdf.geojson
                    common.config_to_file(config_gdf, ROI_path)
                print("Saved config files for each downloaded ROI")
            elif is_downloaded == False:
                # if data is not downloaded save to current working directory
                filepath = os.path.abspath(os.getcwd())
                # save to config.json
                common.config_to_file(config_json, filepath)
                # save to config_gdf.geojson
                common.config_to_file(config_gdf, filepath)
                print("Saved config files for each ROI.")

    def save_settings(self, **kwargs):
        """Saves the settings for downloading data in a dictionary
        Pass in data in the form of
        save_settings(sat_list=sat_list, dates=dates,**preprocess_settings)
        *You must use the names sat_list, landsat_collection, and dates
        """
        tmp_settings = self.settings.copy()
        for key, value in kwargs.items():
            tmp_settings[key] = value

        self.settings = tmp_settings.copy()
        del tmp_settings
        logger.info(f"Settings: {self.settings}")

    def roi_on_hover(self, feature, **kwargs):
        # Modifies html of accordion when roi is hovered over
        values = defaultdict(lambda: "unknown", feature["properties"])
        # convert area of ROI to km^2
        roi_area = common.get_area(feature["geometry"]) * 10**-6
        self.accordion.children[
            0
        ].value = """
        <h2>ROI</h2>
        <p>Id: {}</p>
        <p>Area(km²): {}</p>
        """.format(
            values["id"], roi_area
        )

    def get_selected_rois(self, roi_ids: list) -> gpd.GeoDataFrame:
        """Returns a geodataframe of all rois whose ids are in given list
        roi_ids.

        Args:
            roi_ids (list[str]): ids of ROIs

        Returns:
            gpd.GeoDataFrame:  geodataframe of all rois selected by the roi_ids
        """
        selected_rois_gdf = self.rois.gdf[self.rois.gdf["id"].isin(roi_ids)]
        return selected_rois_gdf

    def remove_all(self) -> None:
        """Remove the rois and imagery from the map"""
        self.remove_all_rois()
        self.remove_segmentation_layers()

    def remove_segmentation_layers(self) -> None:
        """
        Removes segmentation and original layers from the map and clears related lists.

        This function loops through the segmentation and original layers and removes them from the map if they are present.
        It also clears the `self.original_layers`, `self.seg_layers`, and `self.years` lists.

        Args:
            None

        Returns:
            None
        """
        # Remove all original and segmentation layers from the map
        remove_layers = [
            layer
            for layer in self.original_layers + self.seg_layers
            if layer in self.map.layers
        ]
        logger.info(f"remove_layers: {list(map(lambda x:x.name,remove_layers))}")
        for layer in remove_layers:
            self.map.remove(layer)

        # Clear the lists of original and segmentation layers and years
        self.original_layers.clear()
        self.seg_layers.clear()
        self.years.clear()

    def remove_layer_by_name(self, layer_name: str):
        existing_layer = self.map.find_layer(layer_name)
        if existing_layer is not None:
            self.map.remove(existing_layer)
        logger.info(f"Removed layer {layer_name}")

    def replace_layer_by_name(
        self, layer_name: str, new_layer: GeoJSON, on_hover=None, on_click=None
    ) -> None:
        """Replaces layer with layer_name with new_layer on map. Adds on_hover and on_click callable functions
        as handlers for hover and click events on new_layer
        Args:
            layer_name (str): name of layer to replace
            new_layer (GeoJSON): ipyleaflet GeoJSON layer to add to map
            on_hover (callable, optional): Callback function that will be called on hover event on a feature, this function
            should take the event and the feature as inputs. Defaults to None.
            on_click (callable, optional): Callback function that will be called on click event on a feature, this function
            should take the event and the feature as inputs. Defaults to None.
        """
        logger.info(
            f"layer_name {layer_name} \non_hover {on_hover}\n on_click {on_click}"
        )
        self.remove_layer_by_name(layer_name)
        exception_handler.check_empty_layer(new_layer, layer_name)
        # when feature is hovered over on_hover function is called
        if on_hover is not None:
            new_layer.on_hover(on_hover)
        if on_click is not None:
            new_layer.on_click(on_click)
        self.map.add_layer(new_layer)
        logger.info(f"Add layer to map: {layer_name}")

    def remove_selected_rois(self, ids: List[int]) -> None:
        """Remove selected regions of interest (ROIs).

        Removes the ROIs with the specified `ids` from the `self.rois` GeoDataFrame. The `ids` are also
        removed from the `self.selected_set` set of selected ROIs.

        Logs information about the `self.rois.gdf`, `self.selected_set`, and the remaining ROI IDs after
        removing the specified `ids`.

        Args:
            ids (List[int]): A list of integers representing the IDs of the ROIs to remove.

        Returns:
            None
        """
        self.rois.remove_ids(set(ids))
        self.selected_set -= set(ids)
        logger.info(f"self.rois.gdf after removing ids {ids}: {self.rois.gdf}")
        logger.info(f"selected_set {self.selected_set} after removing ids {ids}")
        logger.info(f"IDs {self.rois.get_ids()} after removing ids {ids}")

    def remove_all_rois(self) -> None:
        """Removes all the unselected rois from the map"""
        self.draw_control.clear()
        existing_layer = self.map.find_layer(ROI.LAYER_NAME)
        if existing_layer is not None:
            self.map.remove(existing_layer)
        self.rois = ROI()
        logger.info("Removing all ROIs from map")
        # Remove the selected and unselected rois
        self.remove_layer_by_name(self.SELECTED_LAYER_NAME)
        self.remove_layer_by_name(ROI.LAYER_NAME)
        # clear all the ids from the selected set
        self.selected_set = set()

    def create_DrawControl(self, draw_control: "ipyleaflet.leaflet.DrawControl"):
        """Modifies given draw control so that only rectangles can be drawn

        Args:
            draw_control (ipyleaflet.leaflet.DrawControl): draw control to modify

        Returns:
            ipyleaflet.leaflet.DrawControl: modified draw control with only ability to draw rectangles
        """
        draw_control.polyline = {}
        draw_control.circlemarker = {}
        draw_control.polygon = {}
        draw_control.rectangle = {
            "shapeOptions": {
                "fillColor": "green",
                "color": "green",
                "fillOpacity": 0.1,
                "Opacity": 0.1,
            },
            "drawError": {"color": "#dd253b", "message": "Ops!"},
            "allowIntersection": False,
            "transform": True,
        }
        return draw_control

    def exit_delete_state(self):
        """Exit delete state for the map.

        This function removes the ROI layer with the "select for delete" on click handlers attached,
        replaces it with a new ROI layer that has the "select" on click handler, and brings the
        selected ROI layer back onto the map with the deleted ROIs removed. It also empties the
        delete set and removes the layer containing the ROIs selected for deletion.

        Args:
            None

        Returns:
            None
        """
        # remove the roi layer that has the select for delete on click handlers attached
        self.remove_layer_by_name(ROI.LAYER_NAME)
        # replaces roi layer new layer that has on click handler to select the roi
        if not self.rois.gdf.empty:
            # add roi hover and click handlers for roi
            on_hover = self.roi_on_hover
            on_click = self.select_onclick_handler
            layer_name = ROI.LAYER_NAME
            # load new roi layer on map that doesn't contain rois that were removed
            self.load_on_map(self.rois, layer_name, on_hover, on_click)
        # bring selected roi layer back onto map with deleted rois removed
        if len(self.selected_set) > 0:
            layer_name = ROI.LAYER_NAME
            # create new selected layer with remaining ids in selected set
            selected_layer = GeoJSON(
                data=self.convert_selected_set_to_geojson(
                    self.selected_set, layer_name=layer_name
                ),
                name=self.SELECTED_LAYER_NAME,
                hover_style={"fillColor": "blue", "fillOpacity": 0.1, "color": "aqua"},
            )
            logger.info(f"selected_layer: {selected_layer}")
            # add new selected layer onto map
            self.replace_layer_by_name(
                self.SELECTED_LAYER_NAME,
                selected_layer,
                on_click=self.deselect_onclick_handler,
                on_hover=self.roi_on_hover,
            )
        # empty self.delete_set so its empty for next time
        self.delete_set = set()
        # remove layer containing rois selected for deletion
        self.remove_layer_by_name(self.DELETE_LAYER_NAME)

    def enter_delete_state(self):
        """Enter delete state for the map.

        This function removes the selected layer from the map and changes the color of
        the ROIs (Regions of Interest) to red to indicate that they will be deleted. It also
        sets the on_hover and on_click properties for the ROIs to update the ROI HTML and allows
        the user to select the ROIs for deletion, respectively.

        Args:
            None

        Returns:
            None
        """
        # if no ROIs are on the map return
        if self.map.find_layer(ROI.LAYER_NAME) is None:
            return
        # temporarily remove selected layer from map
        self.remove_layer_by_name(self.SELECTED_LAYER_NAME)
        # show roi area on hover
        on_hover = self.roi_on_hover
        # use ROI layer's "on click" event to color ROIs red to indicate they will be deleted
        on_click = self.select_for_delete_onclick
        layer_name = ROI.LAYER_NAME
        self.load_on_map(self.rois, layer_name, on_hover, on_click)

    def launch_delete_box(
        self, row: "ipywidgets.HBox", title: str = None, msg: str = None
    ) -> None:
        """Launch the delete box for the map.

        This function displays a delete box to the user, puts the ROI layers in delete state,
        and adds the delete box to a row in the user interface.

        Args:
            row: A row widget (instance of `ipywidgets.HBox`) to which the delete box will be added.
            title: An optional title for the delete box.
            msg: An optional message for the delete box.

        Returns:
            None
        """
        # Show user error message
        delete_box = self.create_delete_box(title=title, msg=msg)
        # put roi layers in delete state where they will turn red on click and add ids to delete set
        self.enter_delete_state()
        # clear row and close all widgets in self.file_row before adding new delete_box
        common.clear_row(row)
        # add instance of delete_box to row
        row.children = [delete_box]

    def handle_draw(
        self, target: "ipyleaflet.leaflet.DrawControl", action: str, geo_json: dict
    ):
        """Adds or removes the roi  when drawn/deleted from map
        Args:
            target (ipyleaflet.leaflet.DrawControl): draw control used
            action (str): name of the most recent action ex. 'created', 'deleted'
            geo_json (dict): geojson dictionary
        """
        logger.info("Draw occurred")
        if (
            self.draw_control.last_action == "created"
            and self.draw_control.last_draw["geometry"]["type"] == "Polygon"
        ):
            try:
                geometry = self.draw_control.last_draw["geometry"]
                self.rois.add_geometry(geometry)
                self.load_feature_on_map()
            except exceptions.TooLargeError as too_big:
                exception_handler.handle_bbox_error(too_big, self.warning_box)
            except exceptions.TooSmallError as too_small:
                exception_handler.handle_bbox_error(too_small, self.warning_box)
            except Exception as error:
                exception_handler.handle_exception(error, self.warning_box)
            finally:
                self.draw_control.clear()

    def load_feature_on_map(
        self,
        file: str = "",
        **kwargs,
    ) -> None:
        """Load a feature on a map.

        Loads a GeoDataFrame as a feature on a map. If a `file` is provided, the function reads the
        GeoDataFrame from the file. If a `gdf` is provided, the function uses it directly. If neither
        is provided, it uses the GeoDataFrame stored in `self.rois.gdf`.

        The feature is added to the map using the `self.load_on_map` method, with the `layer_name`
        set to `ROI.LAYER_NAME`, the `on_hover` set to `self.roi_on_hover`, and the `on_click` set
        to `self.select_onclick_handler`.

        Args:
            file (str, optional): The file path to read the GeoDataFrame from. Default is an empty string.
            gdf (gpd.GeoDataFrame, optional): The GeoDataFrame to use. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        # if file is passed read gdf from file
        if file != "":
            logger.info(f"Loading file on map: {file}")
            gdf = common.read_gpd_file(file)
            exception_handler.check_if_gdf_empty(gdf, "roi")
            self.rois.add_geodataframe(gdf)
            bounds = gdf.total_bounds
            self.map.zoom_to_bounds(bounds)

        new_feature = self.rois
        on_hover = self.roi_on_hover
        on_click = self.select_onclick_handler
        layer_name = ROI.LAYER_NAME
        # load new feature on map
        self.load_on_map(new_feature, layer_name, on_hover, on_click)

    def load_on_map(
        self, feature, layer_name: str, on_hover=None, on_click=None
    ) -> None:
        """Loads feature on map

        Replaces current feature layer on map with given feature

        Raises:
            Exception: raised if feature layer is empty
        """
        # style and add the feature to the map
        new_layer = self.create_layer(feature, layer_name)
        # Replace old feature layer with new feature layer
        self.replace_layer_by_name(
            layer_name, new_layer, on_hover=on_hover, on_click=on_click
        )

    def create_layer(self, feature, layer_name: str) -> "ipyleaflet.GeoJSON":
        """Create a layer from a feature.

        Creates a layer from a feature for display on a map. If the feature has an attribute `gdf`,
        the layer is created from the GeoDataFrame stored in `gdf`. If the `gdf` is empty, a warning
        is logged and `None` is returned. If the feature does not have an attribute `gdf`, an exception
        is raised.

        The layer is styled using the `feature.style_layer` method.

        Args:
            feature (object): The feature to create the layer from.
            layer_name (str): The name of the layer.

        Returns:
            ipyleaflet.GeoJSON: The styled layer, represented as a GeoJSON object.

        Raises:
            Exception: If the feature is invalid or does not have an attribute `gdf`.
        """
        logger.info(f"Creating layer from feature: {feature}")
        logger.info(f"Creating layer from feature.gdf: {feature.gdf}")
        if hasattr(feature, "gdf"):
            if feature.gdf.empty:
                logger.warning("Cannot add an empty geodataframe layer to the map.")
                print("Cannot add an empty geodataframe layer to the map.")
                return None
            layer_geojson = json.loads(feature.gdf.to_json())
        elif not hasattr(feature, "gdf"):
            raise Exception(
                f"Invalid feature or no feature provided. Cannot create layer.{type(feature)}"
            )
        # convert layer to GeoJson and style it accordingly
        styled_layer = feature.style_layer(layer_geojson, layer_name)
        return styled_layer

    def select_for_delete_onclick(
        self, event: str = None, id: "NoneType" = None, properties: dict = None, **args
    ):
        """On click handler for when unselected geojson is clicked.

        Adds feature's id to delete_set. Replaces current selected layer with a new one that includes
        recently clicked geojson.

        Args:
            event (str, optional): event fired ('click'). Defaults to None.
            id (NoneType, optional):  Defaults to None.
            properties (dict, optional): geojson dict for clicked geojson. Defaults to None.
        """
        if properties is None:
            return
        logger.info(f"properties of roi selected for deletion : {properties}")
        logger.info(f"ROI_id of roi selected for deletion : {properties['id']}")

        # Add id of clicked ROI to selected_set
        ROI_id = str(properties["id"])
        self.delete_set.add(ROI_id)
        logger.info(f"Added ID to delete_set: {self.delete_set}")
        # create new layer of rois selected for deletion
        layer_name = ROI.LAYER_NAME
        delete_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(
                self.delete_set,
                layer_name=layer_name,
                color="red",
            ),
            name=self.DELETE_LAYER_NAME,
            hover_style={"fillColor": "red", "fillOpacity": 0.1, "color": "red"},
        )
        logger.info(f"delete_layer: {delete_layer}")
        self.replace_layer_by_name(
            self.DELETE_LAYER_NAME,
            delete_layer,
            on_click=self.deselect_for_delete_onclick,
            on_hover=self.roi_on_hover,
        )

    def deselect_for_delete_onclick(
        self, event: str = None, id: "NoneType" = None, properties: dict = None, **args
    ):
        """On click handler for rois selected for deletion layer.

        Removes clicked roi's id from the deleted_set and replaces the delete layer with a new one with
        the clicked roi removed from delete_layer.

        Args:
            event (str, optional): event fired ('click'). Defaults to None.
            id (NoneType, optional):  Defaults to None.
            properties (dict, optional): geojson dict for clicked selected geojson. Defaults to None.
        """
        if properties is None:
            return
        # Remove the current layers cid from selected set
        logger.info(f"properties of ROI selected: {properties}")
        logger.info(f"ROI_id to remove from delete set: {properties['id']}")
        roi_id = properties["id"]
        self.delete_set.remove(roi_id)
        logger.info(f"delete set after ID removal: {self.delete_set}")
        # Recreate delete layers without layer that was removed
        layer_name = ROI.LAYER_NAME
        delete_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(
                self.delete_set,
                layer_name=layer_name,
                color="red",
            ),
            name=self.DELETE_LAYER_NAME,
            hover_style={"fillColor": "red", "fillOpacity": 0.1, "color": "red"},
        )
        self.replace_layer_by_name(
            self.DELETE_LAYER_NAME,
            delete_layer,
            on_click=self.select_for_delete_onclick,
            on_hover=self.roi_on_hover,
        )

    def select_onclick_handler(
        self, event: str = None, id: "NoneType" = None, properties: dict = None, **args
    ):
        """On click handler for when unselected geojson is clicked.

        Adds feature's id to selected_set. Replaces current selected layer with a new one that includes
        recently clicked geojson.

        Args:
            event (str, optional): event fired ('click'). Defaults to None.
            id (NoneType, optional):  Defaults to None.
            properties (dict, optional): geojson dict for clicked geojson. Defaults to None.
        """
        if properties is None:
            return
        logger.info(f"properties : {properties}")
        logger.info(f"ROI_id : {properties['id']}")

        # Add id of clicked ROI to selected_set
        ROI_id = str(properties["id"])
        self.selected_set.add(ROI_id)
        logger.info(f"Added ID to selected set: {self.selected_set}")
        # create new layer of selected rois
        layer_name = ROI.LAYER_NAME
        selected_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(
                self.selected_set, layer_name=layer_name
            ),
            name=self.SELECTED_LAYER_NAME,
            hover_style={"fillColor": "blue", "fillOpacity": 0.1, "color": "aqua"},
        )
        logger.info(f"selected_layer: {selected_layer}")
        self.replace_layer_by_name(
            self.SELECTED_LAYER_NAME,
            selected_layer,
            on_click=self.deselect_onclick_handler,
            on_hover=self.roi_on_hover,
        )

    def deselect_onclick_handler(
        self, event: str = None, id: "NoneType" = None, properties: dict = None, **args
    ):
        """On click handler for selected geojson layer.

        Removes clicked layer's cid from the selected_set and replaces the select layer with a new one with
        the clicked layer removed from select_layer.

        Args:
            event (str, optional): event fired ('click'). Defaults to None.
            id (NoneType, optional):  Defaults to None.
            properties (dict, optional): geojson dict for clicked selected geojson. Defaults to None.
        """
        if properties is None:
            return
        # Remove the current layers cid from selected set
        logger.info(f"deselect_onclick_handler: properties : {properties}")
        logger.info(f"deselect_onclick_handler: ROI_id to remove : {properties['id']}")
        cid = properties["id"]
        self.selected_set.remove(cid)
        logger.info(f"selected set after ID removal: {self.selected_set}")
        self.remove_layer_by_name(self.SELECTED_LAYER_NAME)
        # Recreate selected layers without layer that was removed
        layer_name = ROI.LAYER_NAME
        selected_layer = GeoJSON(
            data=self.convert_selected_set_to_geojson(
                self.selected_set, layer_name=layer_name
            ),
            name=self.SELECTED_LAYER_NAME,
            hover_style={"fillColor": "blue", "fillOpacity": 0.1, "color": "aqua"},
        )
        self.replace_layer_by_name(
            self.SELECTED_LAYER_NAME,
            selected_layer,
            on_click=self.deselect_onclick_handler,
            on_hover=self.roi_on_hover,
        )

    def save_feature_to_file(self, feature: ROI, filename: str = "ROI.geojson"):
        if isinstance(feature, ROI):
            if feature.gdf.empty:
                logger.error(f"No ROIs loaded on the map. {feature.gdf}")
                raise Exception("No ROIs loaded on the map")
            # raise exception if no rois were selected
            exception_handler.check_selected_set(self.selected_set)
            logger.info(
                f"feature.gdf.loc[self.selected_set]{feature.gdf.loc[self.selected_set]}"
            )
            feature.gdf.loc[self.selected_set].to_file(filename, driver="GeoJSON")
        print(f"Save {feature.LAYER_NAME} to {filename}")
        logger.info(f"Save {feature.LAYER_NAME} to {filename}")

    def convert_selected_set_to_geojson(
        self, selected_set: set, layer_name: str = "", color: str = "blue"
    ) -> dict:
        """Returns a geojson dictionary containing features with ids in selected set. Features are styled with a blue border and
        blue fill color to indicate they've been selected by the user.
        Args:
            selected_set (set): ids of selected geojson
            layer_name (str): name of roi layer containing all rois
        Returns:
           dict: geojson dict containing FeatureCollection for all geojson objects in selected_set
        """
        # create a new geojson dictionary to hold selected ROIs
        selected_rois = {"type": "FeatureCollection", "features": []}
        roi_layer = self.map.find_layer(layer_name)
        logger.info(f"roi_layer: {roi_layer}")
        # if ROI layer does not exist throw an error
        exception_handler.check_empty_layer(roi_layer, layer_name)
        # Copy only selected ROIs with id in selected_set
        selected_rois["features"] = [
            feature
            for feature in roi_layer.data["features"]
            if feature["properties"]["id"] in selected_set
        ]
        logger.info(f"selected_rois['features']: {selected_rois['features']}")
        # Each selected ROI will be blue and unselected rois will appear black
        for feature in selected_rois["features"]:
            feature["properties"]["style"] = {
                "color": color,
                "weight": 2,
                "fillColor": color,
                "fillOpacity": 0.1,
            }
        return selected_rois
