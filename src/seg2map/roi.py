# Standard library imports
import logging
from typing import Union
from functools import lru_cache

# Internal dependencies imports
from .exceptions import TooLargeError, TooSmallError
from seg2map import common

# External dependencies imports
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from ipyleaflet import GeoJSON


logger = logging.getLogger(__name__)


class ROI:
    """ROI

    A Bounding Box drawn by user.
    """

    MAX_AREA = 100000000  # UNITS = Sq. Meters
    MIN_AREA = 10  # UNITS = Sq. Meters
    LAYER_NAME = "ROI"

    def __init__(
        self,
    ):
        self.gdf = gpd.GeoDataFrame()
        self.settings = {}
        # self.settings={"sitename":"","filepath":"","ids":[],"dates":""}
        self.filename = "roi.geojson"

    @lru_cache()
    def get_ids(self) -> list:
        return self.gdf.index.to_list()

    def set_settings(self, roi_settings: dict):
        logger.info(f"Old settings: {self.get_settings()}")
        logger.info(f"New settings to replace old settings: {roi_settings}")
        self.settings = roi_settings
        logger.info(f"New Settings: {self.settings}")

    def get_settings(self) -> dict:
        return self.settings

    def create_geodataframe_from_geometry(
        self, rectangle: dict, new_id: str = "", crs: str = "EPSG:4326"
    ) -> gpd.GeoDataFrame:
        """Creates a geodataframe with the crs specified by crs
        Args:
            rectangle (dict): geojson dictionary
            crs (str, optional): coordinate reference system string. Defaults to 'EPSG:4326'.

        Returns:
            gpd.GeoDataFrame: geodataframe with geometry column = rectangle and given crs"""
        geom = [shape(rectangle)]
        gdf = gpd.GeoDataFrame({"geometry": geom})
        gdf.crs = crs
        gdf["id"] = new_id
        gdf.index = gdf["id"]
        gdf.index = gdf.index.rename("ROI_ID")
        logger.info(f"new geodataframe created: {gdf}")
        return gdf

    def add_geometry(self, geometry: dict, crs: str = "EPSG:4326"):
        """
        Add a new geometry to the main geodataframe.

        Parameters:
        - geometry (dict): The new geometry to be added, represented as a dictionary.
        - crs (str): The Coordinate Reference System (CRS) of the geometry, with a default value of "EPSG:4326".

        Raises:
        - TypeError: If the `geometry` argument is not of type dict.

        Returns:
        - None

        """
        logger.info(f"geometry: {geometry}")
        if not isinstance(geometry, dict):
            logger.error(
                f"TypeError: Expected argument of type int, got {type(geometry)}"
            )
            raise TypeError(
                "Expected argument of type int, got {}".format(type(geometry))
            )
        bbox_area = common.get_area(geometry)
        ROI.check_size(bbox_area)
        # create id for new geometry
        new_id = common.generate_random_string(self.get_ids())
        # create geodataframe from geometry
        new_gdf = self.create_geodataframe_from_geometry(geometry, new_id, crs)
        # add geodataframe to main geodataframe
        self.gdf = self.add_new(new_gdf)
        logger.info(f"Add geometry: {geometry}\n self.gdf {self.gdf}")

    def add_geodataframe(
        self, new_gdf: gpd.GeoDataFrame, crs: str = "EPSG:4326"
    ) -> None:
        # check if geodataframe column has 'id' column and add one if one doesn't exist
        if "id" not in new_gdf.columns:
            logger.info("Id not in columns.")
            # none of the new ids can already exist in self.gdf
            avoid_list = self.gdf.index.to_list()
            ids = []
            # generate a new id for each ROI in the geodataframe
            for _ in range(len(new_gdf)):
                new_id = common.generate_random_string(avoid_list)
                ids.append(new_id)
                avoid_list.append(new_id)
            logger.info(f"Adding IDs{ids}")
            new_gdf["id"] = ids
            new_gdf.index = new_gdf["id"]
            new_gdf.index = new_gdf.index.rename("ROI_ID")

            logger.info(f"New gdf after adding IDs: {new_gdf}")

        # get row ids of ROIs whose area exceeds MAX AREA
        drop_ids = common.get_ids_with_invalid_area(new_gdf, max_area=ROI.MAX_AREA)
        if len(drop_ids) > 0:
            print("Dropping ROIs that are an invalid size ")
            logger.info(f"Dropping ROIs that are an invalid size {drop_ids}")
            new_gdf.drop(index=drop_ids, axis=0, inplace=True)
        # convert crs of ROIs to the map crs
        new_gdf.to_crs(crs)
        new_gdf.index = new_gdf["id"]
        new_gdf.index = new_gdf.index.rename("ROI_ID")
        # add new_gdf to self.gdf
        self.gdf = self.add_new(new_gdf)
        logger.info(f"self.gdf: {self.gdf}")

    def add_new(
        self,
        new_gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Adds a new roi entry to self.gdf.
        Args:
            new_gdf (geodataframe): new ROI to add
        Returns:
            gpd.GeoDataFrame: geodataframe with new roi added to it"""
        # concatenate new geodataframe to existing gdf of rois
        logger.info(f"self.gdf: {self.gdf}")
        logger.info(f"Adding gdf: {new_gdf}")
        new_gdf = gpd.GeoDataFrame(pd.concat([self.gdf, new_gdf], ignore_index=False))
        return new_gdf

    def get_geodataframe(self) -> gpd.GeoDataFrame:
        return self.gdf

    # def add_new(
    #     self, new_gdf: gpd.GeoDataFrame,
    # ) -> gpd.GeoDataFrame:
    #     """Adds a new roi entry to self.gdf.
    #     Args:
    #         new_gdf (geodataframe): new ROI to add
    #     Returns:
    #         gpd.GeoDataFrame: geodataframe with new roi added to it"""
    #     # concatenate new geodataframe to existing gdf of rois
    #     new_gdf = gpd.GeoDataFrame(pd.concat([self.gdf, new_gdf], ignore_index=True))
    #     return self

    # def add_new(
    #     self, geometry: dict, id: str = "", crs: str = "EPSG:4326"
    # ) -> gpd.GeoDataFrame:
    #     """Adds a new roi entry to self.gdf. New roi has given geometry and a column called
    #     "id" with id.
    #     Args:
    #         geometry (dict): geojson dictionary of roi shape
    #         id(str): unique id of roi
    #         crs (str, optional): coordinate reference system string. Defaults to 'EPSG:4326'.

    #     Returns:
    #         gpd.GeoDataFrame: geodataframe with new roi added to it"""
    #     # create new geodataframe with geomtry and id
    #     geom = [shape(geometry)]
    #     new_roi = gpd.GeoDataFrame({"geometry": geom})
    #     new_roi["id"] = id
    #     new_roi.crs = crs
    #     # concatenate new geodataframe to existing gdf of rois
    #     new_gdf = gpd.GeoDataFrame(pd.concat([self.gdf, new_roi], ignore_index=True))
    #     # update self gdf to have new roi
    #     self.gdf = new_gdf
    #     return self

    # def remove_by_id(
    #     self, roi_id: str = "", crs: str = "EPSG:4326"
    # ) -> gpd.GeoDataFrame:
    #     """Removes roi with id matching roi_id from self.gdf
    #     Args:
    #         roi_id(str): unique id of roi to remove
    #         crs (str, optional): coordinate reference system string. Defaults to 'EPSG:4326'.

    #     Returns:
    #         gpd.GeoDataFrame: geodataframe without roi roi_id in it"""
    #     # create new geodataframe with geomtry and roi_id
    #     new_gdf = self.gdf[self.gdf["id"] != roi_id]
    #     # update self gdf to have new roi
    #     self.gdf = new_gdf
    #     return new_gdf

    def remove_ids(self, roi_id: Union[str, set] = "") -> None:
        """Removes roi with id matching roi_id from self.gdf
        Args:
            roi_id(str): unique id of roi to remove"""
        logger.info(f"Dropping IDs: {roi_id}")
        if isinstance(roi_id, set):
            self.gdf.drop(roi_id, inplace=True)
        else:
            if roi_id in self.gdf.index:
                self.gdf.drop(roi_id, inplace=True)
        logger.info(f"ROI.index after drop: {self.gdf.index}")

    def style_layer(self, geojson: dict, layer_name: str) -> "ipyleaflet.GeoJSON":
        """Return styled GeoJson object with layer name

        Args:
            geojson (dict): geojson dictionary to be styled
            layer_name(str): name of the GeoJSON layer
        Returns:
            "ipyleaflet.GeoJSON": ROIs as GeoJson layer styled as black box that turn
            yellow on hover
        """
        assert geojson != {}, "ERROR.\n Empty geojson cannot be drawn onto map"
        return GeoJSON(
            data=geojson,
            name=layer_name,
            style={
                "color": "#555555",
                "fill_color": "#555555",
                "fillOpacity": 0.1,
                "weight": 1,
            },
            hover_style={"color": "yellow", "fillOpacity": 0.1, "color": "yellow"},
        )

    def check_size(box_area: float):
        """ "Raises an exception if the size of the bounding box is too large or small."""
        # Check if the size is greater than MAX_SIZE
        if box_area > ROI.MAX_AREA:
            raise TooLargeError()
        # Check if size smaller than MIN_SIZE
        elif box_area < ROI.MIN_AREA:
            raise TooSmallError()
