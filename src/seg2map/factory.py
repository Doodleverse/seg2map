import logging
from typing import Union

from src.seg2map.roi import ROI
from src.seg2map import exception_handler

logger = logging.getLogger(__name__)


class Factory:
    def _get_feature_maker(self, feature_type):
        """Returns a callable function for creating a specific type of feature.
         If the feature_type is neither of these, it returns None.
        Args:
            feature_type (str): The type of feature to create. Valid values are 'rois'.

        Returns:
            Callable: A function for creating the specified type of feature.
        """
        logger.info(f"feature_type: {feature_type}")
        # Return the appropriate function based on the feature type
        if "roi" in feature_type.lower():
            return create_rois
        else:
            logger.error(f"Invalid feature type given {feature_type} is not recognized")
            raise Exception(
                f"Invalid feature type given {feature_type} is not recognized"
            )

    def make_feature(
        self,
        coastsegmap: "CoastSeg_Map",
        feature_name: str,
        gdf: "geodataframe" = None,
        **kwargs,
    ) -> ROI:
        """creates a feature of type feature_name from a geodataframe given by gdf or
        from scratch if no geodataframe is provided. If no geodataframe is provided and
        a roi exists within instance of coastsegmap the feature is created within
        the roi.

        Args:
            coastsegmap (CoastSeg_Map): instance of coastsegmap
            feature_name (str): name of feature type to create must be one of the following
            "rois"
            gdf (geodataframe): geodataframe to create feature from
        Returns:
            ROI: new feature created in coastsegmap
        """
        logger.info(f"kwargs: {kwargs}")
        # get function to create feature based on feature requested
        feature_maker = self._get_feature_maker(feature_name)
        # create feature from geodataframe given by gdf
        if gdf is not None:
            return feature_maker(coastsegmap, gdf)
        # if geodataframe not provided then create feature from scratch
        return feature_maker(coastsegmap, **kwargs)


def create_rois(coastsegmap, gdf: "geopandas.GeoDataFrame" = None, **kwargs) -> ROI:
    """Creates roi using last drawn shape on map if no gdf(geodataframe) is provided. If a gdf(geodataframe) is
    provided a roi is created from it.

    Sets coastsegmap.roi to created roi
    Raises:
        exceptions.Object_Not_Found: raised if roi is empty
    """
    logger.info(f"kwargs: {kwargs}")
    if "new_id" in kwargs:
        new_id = kwargs["new_id"]
        logger.info(f"new_id: {new_id}")
    logger.info(f"gdf: {gdf}")
    # if gdf is given make a roi from it
    if gdf is not None:
        roi = ROI(gdf)
        exception_handler.check_if_gdf_empty(roi.gdf, "roi")
    else:
        # get last drawn polygon on map and create roi from it
        geometry = coastsegmap.draw_control.last_draw["geometry"]
        logger.info(f"geometry: {geometry}")
        logger.info(f"coastsegmap.rois: {coastsegmap.rois}")
        if coastsegmap.rois is None:
            logger.info("Creating first ROI on map")
            roi = ROI(geometry, new_id)
            logger.info(f"new roi: {roi}")
        else:
            logger.info("Adding another ROI to map")
            # add new roi to existing rois on map
            roi = coastsegmap.rois.add_new(geometry, new_id)
            logger.info(f"new roi: {roi}")
        # make sure roi created is not empty
        exception_handler.check_if_gdf_empty(roi.gdf, "roi")

    # clean drawn feature from map
    coastsegmap.draw_control.clear()
    logger.info("ROI was loaded on map")
    print("ROI was loaded on map")
    coastsegmap.rois = roi
    return roi
