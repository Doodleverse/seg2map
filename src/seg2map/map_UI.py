# standard python imports
import os
import datetime
import logging
from collections import defaultdict

# internal python imports
from src.seg2map import exception_handler

# external python imports
from IPython.display import display
from ipyfilechooser import FileChooser

from google.auth import exceptions as google_auth_exceptions
from ipywidgets import Button
from ipywidgets import ToggleButton
from ipywidgets import HBox
from ipywidgets import VBox
from ipywidgets import Layout
from ipywidgets import DatePicker
from ipywidgets import HTML
from ipywidgets import RadioButtons
from ipywidgets import Text
from ipywidgets import Output


logger = logging.getLogger(__name__)


def create_file_chooser(callback, title: str = None):
    padding = "0px 0px 0px 5px"  # upper, right, bottom, left
    # creates a unique instance of filechooser and button to close filechooser
    geojson_chooser = FileChooser(os.getcwd())
    geojson_chooser.dir_icon = os.sep
    geojson_chooser.filter_pattern = ["*.geojson"]
    geojson_chooser.title = "<b>Select a geojson file</b>"
    if title is not None:
        geojson_chooser.title = f"<b>{title}</b>"
    geojson_chooser.register_callback(callback)

    close_button = ToggleButton(
        value=False,
        tooltip="Close File Chooser",
        icon="times",
        button_style="primary",
        layout=Layout(height="28px", width="28px", padding=padding),
    )

    def close_click(change):
        if change["new"]:
            geojson_chooser.close()
            close_button.close()

    close_button.observe(close_click, "value")
    chooser = HBox([geojson_chooser, close_button], layout=Layout(width="100%"))
    return chooser


class UI:
    # all instances of UI will share the same debug_view
    # this means that UI and coastseg_map must have a 1:1 relationship
    # Output widget used to print messages and exceptions created by CoastSeg_Map
    debug_view = Output(layout={"border": "1px solid black"})
    # Output widget used to print messages and exceptions created by download progress
    download_view = Output(layout={"border": "1px solid black"})
    settings_messages = Output(layout={"border": "1px solid black"})

    def __init__(self, coastseg_map):
        # save an instance of coastseg_map
        self.coastseg_map = coastseg_map
        # button styles
        self.remove_style = dict(button_color="red")
        self.load_style = dict(button_color="#69add1")
        self.action_style = dict(button_color="#ae3cf0")
        self.save_style = dict(button_color="#50bf8f")
        self.clear_stlye = dict(button_color="#a3adac")

        # buttons to load configuration files
        self.load_configs_button = Button(
            description="Load Config", style=self.load_style
        )
        self.load_configs_button.on_click(self.on_load_configs_clicked)
        self.save_config_button = Button(
            description="Save Config", style=self.save_style
        )
        self.save_config_button.on_click(self.on_save_config_clicked)

        self.download_button = Button(
            description="Download Imagery", style=self.action_style
        )
        self.download_button.on_click(self.download_button_clicked)

        # Remove buttons
        self.clear_debug_button = Button(
            description="Clear Messages", style=self.clear_stlye
        )
        self.clear_debug_button.on_click(self.clear_debug_view)

        # Remove buttons
        self.clear_downloads_button = Button(
            description="Clear Downloads", style=self.clear_stlye
        )
        self.clear_downloads_button.on_click(self.clear_download_view)

        # create the HTML widgets containing the instructions
        self._create_HTML_widgets()

    def get_view_settings(self) -> VBox:
        # update settings button
        update_settings_btn = Button(
            description="Update Settings", style=self.action_style
        )
        update_settings_btn.on_click(self.update_settings_btn_clicked)
        self.settings_html = HTML()
        self.settings_html.value = self.get_settings_html(self.coastseg_map.settings)
        view_settings_vbox = VBox([self.settings_html, update_settings_btn])
        return view_settings_vbox

    def get_settings_vbox(self) -> VBox:
        # declare settings widgets
        dates_vbox = self.get_dates_picker()
        self.sitename_field = Text(
            value="sitename",
            placeholder="Type sitename",
            description="Sitename:",
            disabled=False,
        )

        img_type_picker = self.get_image_types_picker()

        settings_button = Button(description="Save Settings", style=self.action_style)
        settings_button.on_click(self.save_settings_clicked)

        # create settings vbox
        settings_vbox = VBox(
            [
                dates_vbox,
                self.sitename_field,
                img_type_picker,
                settings_button,
                UI.settings_messages,
            ]
        )

        return settings_vbox

    def get_dates_picker(self):
        # Date Widgets
        self.start_date = DatePicker(
            description="Start Date",
            value=datetime.date(2010, 1, 1),
            disabled=False,
        )
        self.end_date = DatePicker(
            description="End Date",
            value=datetime.date(2010, 12, 31),  # 2019, 1, 1
            disabled=False,
        )
        date_instr = HTML(value="<b>Pick a date:</b>", layout=Layout(padding="10px"))
        dates_box = HBox([self.start_date, self.end_date])
        dates_vbox = VBox([date_instr, dates_box])
        return dates_vbox

    def get_image_types_picker(self) -> VBox:
        image_types = ["multiband", "singleband", "both"]
        self.img_type_radio = RadioButtons(
            options=image_types,
            value=image_types[0],
            description="",
            disabled=False,
        )
        instr = HTML(
            value="<b>Pick type of imagery:</b>", layout=Layout(padding="10px")
        )
        img_type_vbox = HBox([instr, self.img_type_radio])
        return img_type_vbox

    def save_to_file_buttons(self):
        # save to file buttons
        save_instr = HTML(
            value="<h2>Save to file</h2>\
                Save feature on the map to a geojson file.\
                <br>Geojson file will be saved to Seg2Map directory.\
            ",
            layout=Layout(padding="0px"),
        )

        self.save_radio = RadioButtons(
            options=[
                "ROI",
            ],
            value="ROI",
            description="",
            disabled=False,
        )

        self.save_button = Button(
            description=f"Save {self.save_radio.value} to file", style=self.save_style
        )
        self.save_button.on_click(self.save_to_file_btn_clicked)

        def save_radio_changed(change):
            self.save_button.description = f"Save {str(change['new'])} to file"

        self.save_radio.observe(save_radio_changed, "value")
        save_vbox = VBox([save_instr, self.save_radio, self.save_button])
        return save_vbox

    def remove_buttons(self):
        # define remove feature radio box button
        remove_instr = HTML(
            value="<h2>Remove Feature from Map</h2>",
            layout=Layout(padding="0px"),
        )
        self.remove_button = Button(description=f"Remove ROIs", style=self.remove_style)
        self.remove_button.on_click(self.remove_feature_from_map)
        # define remove all button
        self.remove_all_button = Button(
            description="Remove all", style=self.remove_style
        )
        self.remove_all_button.on_click(self.remove_all_from_map)
        remove_buttons = VBox(
            [
                remove_instr,
                self.remove_button,
                self.remove_all_button,
            ]
        )
        return remove_buttons

    def get_settings_html(
        self,
        settings: dict,
    ):
        # Modifies setttings html
        default = "unknown"
        keys = [
            "dates",
            "sitename",
            "download_bands",
        ]
        values = defaultdict(lambda: "unknown", settings)
        return """ 
        <h2>Settings</h2>
        <p>dates: {}</p>
        <p>sitename: {}</p>
        <p>download_bands: {}</p>
        """.format(
            values["dates"],
            values["sitename"],
            values["download_bands"],
        )

    def _create_HTML_widgets(self):
        """create HTML widgets that display the instructions.
        widgets created: instr_create_ro, instr_save_roi, instr_load_btns
         instr_download_roi
        """
        self.instr_download_roi = HTML(
            value="<h2><b>Download Imagery</b></h2> \
                <li><b>You must click an ROI on the map before you can download ROIs</b> \
                <li>Scroll past the map to see the download progress \
                </br><h3><b><u>Where is my data?</u></b></br></h3> \
                <li>The data you downloaded will be in the 'data' folder in the main CoastSeg directory</li>\
                Each ROI you downloaded will have its own folder with the ROI's ID and\
                </br>the time it was downloaded in the folder name\
                </br><b>Example</b>: 'ID_1_datetime11-03-22__02_33_22'</li>\
                ",
            layout=Layout(margin="0px 0px 0px 5px"),
        )

        self.instr_config_btns = HTML(
            value="<h2><b>Load and Save Config Files</b></h2>\
                <b>Load Config</b>: Load ROIs from file: 'config_gdf.geojson'\
                <li>'config.json' must be in the same directory as 'config_gdf.geojson'.</li>\
                <b>Save Config</b>: Saves rois, shorelines, transects and bounding box to file: 'config_gdf.geojson'\
                ",
            layout=Layout(margin="0px 5px 0px 5px"),  # top right bottom left
        )

    def create_dashboard(self):
        """creates a dashboard containing all the buttons, instructions and widgets organized together."""
        # create settings controls
        settings_controls = self.get_settings_vbox()
        remove_buttons = self.remove_buttons()
        save_to_file_buttons = self.save_to_file_buttons()

        save_vbox = VBox(
            [
                save_to_file_buttons,
                remove_buttons,
            ]
        )
        config_vbox = VBox(
            [self.instr_config_btns, self.load_configs_button, self.save_config_button]
        )
        download_vbox = VBox(
            [
                self.instr_download_roi,
                self.download_button,
                config_vbox,
            ]
        )

        # Static settings HTML used to show currently loaded settings
        static_settings = self.get_view_settings()

        row_0 = HBox([settings_controls, static_settings])
        row_1 = HBox([save_vbox, download_vbox])
        # in this row prints are rendered with UI.debug_view
        row_3 = HBox([self.clear_debug_button, UI.debug_view])
        self.error_row = HBox([])
        self.row_4 = HBox([])
        row_5 = HBox([self.coastseg_map.map])
        row_6 = HBox([self.clear_downloads_button, UI.download_view])

        return display(
            row_0,
            row_1,
            row_3,
            self.error_row,
            self.row_4,
            row_5,
            row_6,
        )

    @debug_view.capture(clear_output=True)
    def update_settings_btn_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        # Display the settings currently loaded into coastseg_map
        try:
            self.settings_html.value = self.get_settings_html(
                self.coastseg_map.settings
            )
        except Exception as error:
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)

    @debug_view.capture(clear_output=True)
    def load_button_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        self.coastseg_map.map.default_style = {"cursor": "wait"}
        try:
            if "shoreline" in btn.description.lower():
                print("Finding Shoreline")
                self.coastseg_map.load_feature_on_map("shoreline")
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)
        self.coastseg_map.map.default_style = {"cursor": "default"}

    @settings_messages.capture(clear_output=True)
    def save_settings_clicked(self, btn):
        # Save dates selected by user
        image_type = str(self.img_type_radio.value)
        dates = [str(self.start_date.value), str(self.end_date.value)]
        sitename = self.sitename_field.value.replace(" ", "")
        settings = {"dates": dates, "sitename": sitename, "download_bands": image_type}
        dates = [datetime.datetime.strptime(_, "%Y-%m-%d") for _ in dates]
        if dates[1] <= dates[0]:
            print("Dates are not correct chronological order")
            print("Settings not saved")
            return
        # check if sitename path exists and if it does tell user they need a new name
        parent_path = os.path.join(os.getcwd(), "data")
        sitename_path = os.path.join(parent_path, sitename)
        if os.path.exists(sitename_path):
            print(
                f"Sorry this sitename already exists at {sitename_path}\nTry another sitename."
            )
            print("Settings not saved")
            return
        elif not os.path.exists(sitename_path):
            print(f"{sitename} will be created at {sitename_path}")
        try:
            self.coastseg_map.save_settings(**settings)
            self.settings_html.value = self.get_settings_html(
                self.coastseg_map.settings
            )
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)

    @download_view.capture(clear_output=True)
    def download_button_clicked(self, btn):
        UI.download_view.clear_output()
        UI.debug_view.clear_output()
        self.coastseg_map.map.default_style = {"cursor": "wait"}
        UI.debug_view.append_stdout("Scroll down past map to see download progress.")
        try:
            self.download_button.disabled = True
            try:
                self.coastseg_map.download_imagery()
            except Exception as error:
                # renders error message as a box on map
                exception_handler.handle_exception(error, self.coastseg_map.warning_box)
        except google_auth_exceptions.RefreshError as exception:
            print(exception)
            exception_handler.handle_exception(
                error,
                self.coastseg_map.warning_box,
                title="Authentication Error",
                msg="Please authenticate with Google using the cell above: \n Authenticate and Initialize with Google Earth Engine (GEE)",
            )
        self.download_button.disabled = False
        self.coastseg_map.map.default_style = {"cursor": "default"}

    def clear_row(self, row: HBox):
        """close widgets in row/column and clear all children
        Args:
            row (HBox)(VBox): row or column
        """
        for index in range(len(row.children)):
            row.children[index].close()
        row.children = []

    @debug_view.capture(clear_output=True)
    def on_load_configs_clicked(self, button):
        # Prompt user to select a config geojson file
        def load_callback(filechooser: FileChooser) -> None:
            try:
                if filechooser.selected:
                    self.coastseg_map.load_configs(filechooser.selected)
                    self.settings_html.value = self.get_settings_html(
                        self.coastseg_map.settings
                    )
            except Exception as error:
                # renders error message as a box on map
                exception_handler.handle_exception(error, self.coastseg_map.warning_box)

        # create instance of chooser that calls load_callback
        file_chooser = create_file_chooser(load_callback)
        # clear row and close all widgets in row_4 before adding new file_chooser
        self.clear_row(self.row_4)
        # add instance of file_chooser to row 4
        self.row_4.children = [file_chooser]

    @debug_view.capture(clear_output=True)
    def on_save_config_clicked(self, button):
        try:
            print("Save config clicked")
            self.coastseg_map.save_config()
            print("Done!")
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)

    @debug_view.capture(clear_output=True)
    def load_feature_from_file(self, btn):
        # Prompt user to select a geojson file
        def load_callback(filechooser: FileChooser) -> None:
            try:
                if filechooser.selected:
                    if "bbox" in btn.description.lower():
                        print(
                            f"Loading bounding box from file: {os.path.abspath(filechooser.selected)}"
                        )
                        self.coastseg_map.load_feature_on_map(
                            "bbox", os.path.abspath(filechooser.selected)
                        )
                    if "rois" in btn.description.lower():
                        print(
                            f"Loading ROIs from file: {os.path.abspath(filechooser.selected)}"
                        )
                        self.coastseg_map.load_feature_on_map(
                            "rois", os.path.abspath(filechooser.selected)
                        )
            except Exception as error:
                # renders error message as a box on map
                exception_handler.handle_exception(error, self.coastseg_map.warning_box)

        # change title of filechooser based on feature selected
        title = "Select a geojson file"
        # create instance of chooser that calls load_callback
        if "bbox" in btn.description.lower():
            title = "Select bounding box geojson file"
        if "rois" in btn.description.lower():
            title = "Select ROI geojson file"
        # create instance of chooser that calls load_callback
        file_chooser = create_file_chooser(load_callback, title=title)
        # clear row and close all widgets in row_4 before adding new file_chooser
        self.clear_row(self.row_4)
        # add instance of file_chooser to row 4
        self.row_4.children = [file_chooser]

    @debug_view.capture(clear_output=True)
    def remove_feature_from_map(self, btn):
        UI.debug_view.clear_output(wait=True)
        try:
            if "rois" in btn.description.lower():
                print(f"Removing ROIs")
                self.coastseg_map.launch_delete_box(self.coastseg_map.remove_box)
                # self.coastseg_map.remove_all_rois()
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)

    @debug_view.capture(clear_output=True)
    def save_to_file_btn_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        try:
            if "bbox" in btn.description.lower():
                print(f"Saving bounding box to file")
                self.coastseg_map.save_feature_to_file(
                    self.coastseg_map.bbox, "bounding box"
                )
            if "rois" in btn.description.lower():
                print(f"Saving ROIs to file")
                self.coastseg_map.save_feature_to_file(self.coastseg_map.rois, "ROI")
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)

    @debug_view.capture(clear_output=True)
    def remove_all_from_map(self, btn):
        try:
            self.coastseg_map.remove_all()
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.coastseg_map.warning_box)

    def clear_debug_view(self, btn):
        UI.debug_view.clear_output()

    def clear_download_view(self, btn):
        UI.download_view.clear_output()
