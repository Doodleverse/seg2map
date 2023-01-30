class Object_Not_Found(Exception):
    """Object_Not_Found: raised when the feature does not exist
    Args:
        Exception: Inherits from the base exception class
    """

    def __init__(self, feature: str, message=""):
        self.msg = f"No {feature.lower()} found on the map.\n{message}"
        self.feature = feature
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"


class No_Images_Available(Exception):
    """No_Images_Available: raised when nothing can be downloaded
    Args:
        Exception: Inherits from the base exception class
    """

    def __init__(self, feature: str, message=""):
        self.msg = f"No {feature.lower()} found on the map.\n{message}"
        self.feature = feature
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"


class Id_Not_Found(Exception):
    """Id_Not_Found: raised when ROI id does not exist
    Args:
        Exception: Inherits from the base exception class
    """

    def __init__(self, id: int = None, msg="The ROI id does not exist."):
        self.msg = msg
        if id is not None:
            self.msg = f"The ROI id {id} does not exist."
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"


class TooLargeError(Exception):
    """TooLargeError: raised when ROI is larger than MAX_SIZE
    Args:
        Exception: Inherits from the base exception class
    """

    def __init__(self, msg="The ROI was too large."):
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"


class TooSmallError(Exception):
    """TooLargeError: raised when ROI is smaller than MIN_SIZE
    Args:
        Exception: Inherits from the base exception class
    """

    def __init__(self, msg="The ROI was too small."):
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"


class DownloadError(Exception):
    """DownloadError: raised when a download error occurs.
    Args:
        Exception: Inherits from the base exception class
    """

    def __init__(self, file):
        msg = f"\n ERROR\nShoreline file:'{file}' is not online.\nPlease raise an issue on GitHub with the shoreline name.\n https://github.com/SatelliteShorelines/CoastSeg/issues"
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"
