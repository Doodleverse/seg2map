# Standard library imports
import logging
from typing import Union

# Internal dependencies imports
from .exceptions import TooLargeError, TooSmallError
from src.seg2map import common

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

    MAX_AREA = 3000000000  # UNITS = Sq. Meters
    MIN_AREA = 10  # UNITS = Sq. Meters
    LAYER_NAME = "ROI"

    def __init__(
        self,
        rectangle: Union[dict, gpd.GeoDataFrame],
        new_id: str = "",
        filename: str = None,
    ):
        self.gdf = None
        self.settings = {}
        # self.settings={"sitename":"","filepath":"","ids":[],"dates":""}
        self.filename = "roi.geojson"

        if isinstance(rectangle, gpd.GeoDataFrame):
            # check if geodataframe column has 'id' column and add one if one doesn't exist
            if "id" not in rectangle.columns:
                rectangle["id"] = list(map(str, rectangle.index.tolist()))
            # get row ids of ROIs with area that's too large
            drop_ids = common.get_ids_with_invalid_area(
                rectangle, max_area=ROI.MAX_AREA
            )
            if len(drop_ids) > 0:
                print("Dropping ROIs that are an invalid size ")
                logger.info(f"Dropping ROIs that are an invalid size {drop_ids}")
                rectangle.drop(index=drop_ids, axis=0, inplace=True)
            # convert crs of ROIs to the map crs
            rectangle.to_crs("EPSG:4326")
            self.gdf = rectangle

        elif isinstance(rectangle, dict):
            self.gdf = self.create_geodataframe(rectangle, new_id)
        else:
            raise Exception(
                "Invalid rectangle to create ROI. Rectangle must be either a geodataframe or dictionary"
            )
        if filename:
            self.filename = filename

    def set_settings(self, roi_settings: dict):
        self.settings = roi_settings

    def get_settings(self) -> dict:
        return self.settings

    def create_geodataframe(
        self, rectangle: dict, new_id: str = "", crs: str = "EPSG:4326"
    ) -> gpd.GeoDataFrame:
        """Creates a geodataframe with the crs specified by crs
        Args:
            rectangle (dict): geojson dictionary
            crs (str, optional): coordinate reference system string. Defaults to 'EPSG:4326'.

        Returns:
            gpd.GeoDataFrame: geodataframe with geometry column = rectangle and given crs"""
        geom = [shape(rectangle)]
        geojson_bbox = gpd.GeoDataFrame({"geometry": geom})
        geojson_bbox["id"] = new_id
        geojson_bbox.crs = crs
        return geojson_bbox

    def add_new(
        self, geometry: dict, id: str = "", crs: str = "EPSG:4326"
    ) -> gpd.GeoDataFrame:
        """Adds a new roi entry to self.gdf. New roi has given geometry and a column called
        "id" with id.
        Args:
            geometry (dict): geojson dictionary of roi shape
            id(str): unique id of roi
            crs (str, optional): coordinate reference system string. Defaults to 'EPSG:4326'.

        Returns:
            gpd.GeoDataFrame: geodataframe with new roi added to it"""
        # create new geodataframe with geomtry and id
        geom = [shape(geometry)]
        new_roi = gpd.GeoDataFrame({"geometry": geom})
        new_roi["id"] = id
        new_roi.crs = crs
        # concatenate new geodataframe to existing gdf of rois
        new_gdf = gpd.GeoDataFrame(pd.concat([self.gdf, new_roi], ignore_index=True))
        # update self gdf to have new roi
        self.gdf = new_gdf
        return self

    def remove_by_id(
        self, roi_id: str = "", crs: str = "EPSG:4326"
    ) -> gpd.GeoDataFrame:
        """Removes roi with id matching roi_id from self.gdf
        Args:
            roi_id(str): unique id of roi to remove
            crs (str, optional): coordinate reference system string. Defaults to 'EPSG:4326'.

        Returns:
            gpd.GeoDataFrame: geodataframe without roi roi_id in it"""
        # create new geodataframe with geomtry and roi_id
        new_gdf = self.gdf[self.gdf["id"] != roi_id]
        # update self gdf to have new roi
        self.gdf = new_gdf
        return new_gdf

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
