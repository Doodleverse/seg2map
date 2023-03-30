# standard python imports
import os
import datetime
import logging
from collections import defaultdict

# internal python imports
from seg2map import exception_handler
from seg2map import common

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
from ipywidgets import FloatSlider
from ipywidgets import SelectionSlider
from ipywidgets import Dropdown


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
    # this means that UI and seg2map must have a 1:1 relationship
    # Output widget used to print messages and exceptions created by Seg2Map
    debug_view = Output(layout={"border": "1px solid black"})
    # Output widget used to print messages and exceptions created by download progress
    download_view = Output(layout={"border": "1px solid black"})
    settings_messages = Output(layout={"border": "1px solid black"})

    def __init__(self, seg2map):
        # save an instance of Seg2Map
        self.seg2map = seg2map
        # button styles
        self.get_button_styles()

        # buttons to load configuration files
        self.load_configs_button = Button(
            description="Load Config", style=self.load_style
        )
        self.load_configs_button.on_click(self.on_load_configs_clicked)

        # buttons to load configuration files

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

    def get_button_styles(self):
        self.remove_style = dict(button_color="red")
        self.load_style = dict(button_color="#69add1")
        self.action_style = dict(button_color="#ae3cf0")
        self.save_style = dict(button_color="#50bf8f")
        self.clear_stlye = dict(button_color="#a3adac")

    def get_view_settings(self) -> VBox:
        # update settings button
        update_settings_btn = Button(
            description="Update Settings", style=self.action_style
        )
        update_settings_btn.on_click(self.update_settings_btn_clicked)
        self.settings_html = HTML()
        self.settings_html.value = self.get_settings_html(self.seg2map.settings)
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
        settings_button = Button(description="Save Settings", style=self.action_style)
        settings_button.on_click(self.save_settings_clicked)

        # create settings vbox
        settings_vbox = VBox(
            [
                dates_vbox,
                self.sitename_field,
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

    def remove_buttons(self):
        # define remove feature radio box button
        remove_instr = HTML(
            value="<h2>Remove Feature from Map</h2>",
            layout=Layout(padding="0px"),
        )
        self.remove_button = Button(description=f"Remove ROIs", style=self.remove_style)
        self.remove_button.on_click(self.remove_feature_from_map)

        self.remove_seg_button = Button(
            description=f"Remove Imagery", style=self.remove_style
        )
        self.remove_seg_button.on_click(self.remove_seg_clicked)
        # define remove all button
        self.remove_all_button = Button(
            description="Remove all", style=self.remove_style
        )
        self.remove_all_button.on_click(self.remove_all_from_map)
        remove_buttons = VBox(
            [
                remove_instr,
                self.remove_button,
                self.remove_seg_button,
                self.remove_all_button,
            ]
        )
        return remove_buttons

    def segmentation_controls(self):
        # define remove feature radio box button
        instr = HTML(
            value="<h2>Load Segmentations on the Map</h2>",
            layout=Layout(padding="0px"),
        )
        inital_options = ["all"]
        classes = list(self.seg2map.get_classes())
        classes = ["all"] + classes
        if len(self.seg2map.get_classes()) == 0:
            classes = inital_options
        self.class_dropdown = Dropdown(
            options=classes,
            description="Select Class:",
            value=classes[0],
            style={"description_width": "initial"},
        )

        self.load_segmentations_button = Button(
            description="Load Segmentations", style=self.load_style
        )
        self.load_segmentations_button.on_click(self.on_load_session_clicked)

        self.opacity_slider = FloatSlider(
            value=1.0,
            min=0,
            max=1.0,
            step=0.01,
            description="Opacity:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
        )

        default_years = ["2021"]
        years = self.seg2map.get_years()
        if len(years) == 0:
            years = default_years
        self.year_slider = SelectionSlider(
            options=years,
            value=years[0],
            description="Select a year:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
        )

        def year_slider_changed(change):
            year = self.year_slider.value
            seg_layers = self.seg2map.get_seg_layers()
            original_layers = self.seg2map.get_original_layers()
            # order matters: original layers must be before seg layer otherwise so segmentations will appear on to of image
            layers = original_layers + seg_layers
            self.seg2map.load_layers_by_year(layers, year)

        self.year_slider.observe(year_slider_changed, "value")

        def opacity_slider_changed(change):
            # apply opacity to all layers
            year = self.year_slider.value
            opacity = self.opacity_slider.value
            logger.info(f"self.class_dropdown.value: {self.class_dropdown.value}")
            print(f"self.class_dropdown.value: {self.class_dropdown.value}")
            logger.info(f"year: {year}")
            print(f"year: {year}")
            if self.class_dropdown.value == "all":
                seg_layers = self.seg2map.get_seg_layers()
            if self.class_dropdown.value != "all":
                # apply opacity to selected layer name
                seg_layers = self.seg2map.get_seg_layers(self.class_dropdown.value)

            if seg_layers == []:
                return
            self.seg2map.modify_layers_opacity_by_year(seg_layers, year, opacity)

        self.opacity_slider.observe(opacity_slider_changed, "value")

        def handle_class_dropdown(change):
            # apply opacity to all layers
            logger.info(f"handle_class_dropdown: {change['new']}")
            print(f"handle_class_dropdown: {change['new']}")
            if change["new"] == "all":
                year = self.year_slider.value
                seg_layers = self.seg2map.get_seg_layers()
                logger.info(f"seg_layers: {seg_layers}")
                if seg_layers == []:
                    return
                opacity = self.opacity_slider.value
                self.seg2map.modify_layers_opacity_by_year(seg_layers, year, opacity)
            if change["new"] != "all":
                # apply opacity to selected layer name
                year = self.year_slider.value
                seg_layers = self.seg2map.get_seg_layers(change["new"])
                logger.info(f"seg_layers: {seg_layers}")
                if seg_layers == []:
                    return
                opacity = self.opacity_slider.value
                self.seg2map.modify_layers_opacity_by_year(seg_layers, year, opacity)

        self.class_dropdown.observe(handle_class_dropdown, "value")

        opacity_controls = HBox([self.opacity_slider, self.class_dropdown])
        segmentation_box = VBox(
            [
                instr,
                self.load_segmentations_button,
                self.year_slider,
                opacity_controls,
            ]
        )
        return segmentation_box

    def get_settings_html(
        self,
        settings: dict,
    ):
        # Modifies setttings html
        default = "unknown"
        keys = [
            "dates",
            "sitename",
        ]
        values = defaultdict(lambda: "unknown", settings)
        return """ 
        <h2>Settings</h2>
        <p>dates: {}</p>
        <p>sitename: {}</p>
        """.format(
            values["dates"],
            values["sitename"],
        )

    def _create_HTML_widgets(self):
        """create HTML widgets that display the instructions.
        widgets created: instr_create_ro, instr_save_roi, instr_load_btns
         instr_download_roi
        """
        self.instr_download_roi = HTML(
            value="<h2><b>Download Imagery</b></h2> \
                <li><b>You must click an ROI on the map before you can download ROIs</b> \
                <li>The downloaded imagery will be saved to the 'data' directory</li>\
                The folder name for each downloaded ROI will consist of the ROI's ID and the time of download.\
                </br><b>Example</b>: 'ID_1_datetime11-03-22__02_33_22'</li>\
                ",
            layout=Layout(margin="0px 0px 0px 5px"),
        )

        self.instr_config_btns = HTML(
            value="<h2><b>Load and Save Config Files</b></h2>\
                <b>Load Config</b>: Load ROIs from file: 'config_gdf.geojson'\
                <li>'config.json' must be in the same directory as 'config_gdf.geojson'.</li>\
                <b>Save Config</b>: Saves the state of the map to file: 'config_gdf.geojson'\
                ",
            layout=Layout(margin="0px 5px 0px 5px"),  # top right bottom left
        )

    def get_file_controls(self):
        # define remove feature radio box button
        instr = HTML(
            value="<h2>Load & Save GeoJSON Files</h2>",
            layout=Layout(padding="0px"),
        )
        self.load_file_button = Button(
            description=f"Load GeoJSON file",
            icon="fa-file-o",
            style=self.load_style,
        )
        self.load_file_button.on_click(self.load_feature_from_file)

        self.save_button = Button(description=f"Save to GeoJSON", style=self.save_style)
        self.save_button.on_click(self.save_to_file_btn_clicked)
        control_box = VBox(
            [
                instr,
                self.load_file_button,
                self.save_button,
            ]
        )
        return control_box

    def create_dashboard(self):
        """creates a dashboard containing all the buttons, instructions and widgets organized together."""
        # create settings controls
        files_controls = self.get_file_controls()
        settings_controls = self.get_settings_vbox()
        remove_buttons = self.remove_buttons()
        segmentation_controls = self.segmentation_controls()

        self.save_button = Button(
            description=f"Save ROIs to file", style=self.save_style
        )
        self.save_button.on_click(self.save_to_file_btn_clicked)

        save_vbox = VBox([files_controls, remove_buttons, segmentation_controls])
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
        self.file_chooser_row = HBox([])
        row_5 = HBox([self.seg2map.map])
        row_6 = HBox([self.clear_downloads_button, UI.download_view])

        return display(
            row_0,
            row_1,
            row_3,
            self.error_row,
            self.file_chooser_row,
            row_5,
            row_6,
        )

    @debug_view.capture(clear_output=True)
    def update_settings_btn_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        # Display the settings currently loaded into Seg2Map
        try:
            self.settings_html.value = self.get_settings_html(self.seg2map.settings)
        except Exception as error:
            exception_handler.handle_exception(error, self.seg2map.warning_box)

    @settings_messages.capture(clear_output=True)
    def save_settings_clicked(self, btn):
        # Save dates selected by user
        dates = [str(self.start_date.value), str(self.end_date.value)]
        sitename = self.sitename_field.value.replace(" ", "")
        settings = {"dates": dates, "sitename": sitename}
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
            self.seg2map.save_settings(**settings)
            self.settings_html.value = self.get_settings_html(self.seg2map.settings)
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.seg2map.warning_box)

    @download_view.capture(clear_output=True)
    def download_button_clicked(self, btn):
        UI.download_view.clear_output()
        UI.debug_view.clear_output()
        self.seg2map.map.default_style = {"cursor": "wait"}
        print("Scroll down past map to see download progress.")
        try:
            self.download_button.disabled = True
            try:
                self.seg2map.download_imagery()
            except Exception as error:
                # renders error message as a box on map
                exception_handler.handle_exception(error, self.seg2map.warning_box)
        except google_auth_exceptions.RefreshError as exception:
            print(exception)
            exception_handler.handle_exception(
                error,
                self.seg2map.warning_box,
                title="Authentication Error",
                msg="Please authenticate with Google using the cell above: \n Authenticate and Initialize with Google Earth Engine (GEE)",
            )
        self.download_button.disabled = False
        self.seg2map.map.default_style = {"cursor": "default"}

    def clear_row(self, row: HBox):
        """close widgets in row/column and clear all children
        Args:
            row (HBox)(VBox): row or column
        """
        for index in range(len(row.children)):
            row.children[index].close()
        row.children = []

    @debug_view.capture(clear_output=True)
    def on_load_session_clicked(self, button):
        # Prompt user to select a config geojson file
        def load_callback(filechooser: FileChooser) -> None:
            try:
                if filechooser.selected:
                    self.seg2map.load_session(filechooser.selected)
                    years = self.seg2map.years
                    classes = self.seg2map.get_classes()
                    classes = list(classes)
                    if classes:
                        self.class_dropdown.options = ["all"] + classes
                        self.class_dropdown.value = classes[0]
                    if years:
                        self.year_slider.options = years
                        self.year_slider.value = years[0]

            except Exception as error:
                # renders error message as a box on map
                exception_handler.handle_exception(error, self.seg2map.warning_box)

        # create instance of chooser that calls load_callback
        dir_chooser = common.create_dir_chooser(
            load_callback,
            title="Select Session Directory",
            starting_directory="sessions",
        )
        # clear row and close all widgets in row_4 before adding new file_chooser
        self.clear_row(self.file_chooser_row)
        # add instance of file_chooser to row 4
        self.file_chooser_row.children = [dir_chooser]

    @debug_view.capture(clear_output=True)
    def load_segmentations(self, button):
        # Prompt user to select a config geojson file
        def load_callback(filechooser: FileChooser) -> None:
            try:
                if filechooser.selected:
                    self.seg2map.load_configs(filechooser.selected)
                    self.settings_html.value = self.get_settings_html(
                        self.seg2map.settings
                    )
            except Exception as error:
                # renders error message as a box on map
                exception_handler.handle_exception(error, self.seg2map.warning_box)

        # create instance of chooser that calls load_callback
        file_chooser = create_file_chooser(load_callback)
        # clear row and close all widgets in file_chooser_row before adding new file_chooser
        self.clear_row(self.file_chooser_row)
        # add instance of file_chooser to file_chooser_row
        self.file_chooser_row.children = [file_chooser]

    @debug_view.capture(clear_output=True)
    def on_load_configs_clicked(self, button):
        # Prompt user to select a config geojson file
        def load_callback(filechooser: FileChooser) -> None:
            try:
                if filechooser.selected:
                    self.seg2map.load_configs(filechooser.selected)
                    self.settings_html.value = self.get_settings_html(
                        self.seg2map.settings
                    )
            except Exception as error:
                # renders error message as a box on map
                exception_handler.handle_exception(error, self.seg2map.warning_box)

        # create instance of chooser that calls load_callback
        file_chooser = create_file_chooser(load_callback)
        # clear row and close all widgets in file_chooser_row before adding new file_chooser
        self.clear_row(self.file_chooser_row)
        # add instance of file_chooser to file_chooser_row
        self.file_chooser_row.children = [file_chooser]

    @debug_view.capture(clear_output=True)
    def on_save_config_clicked(self, button):
        try:
            self.seg2map.save_config()
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.seg2map.warning_box)

    @debug_view.capture(clear_output=True)
    def remove_feature_from_map(self, btn):
        UI.debug_view.clear_output(wait=True)
        try:
            if "rois" in btn.description.lower():
                self.seg2map.launch_delete_box(self.seg2map.remove_box)
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.seg2map.warning_box)

    @debug_view.capture(clear_output=True)
    def remove_seg_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        try:
            self.seg2map.remove_segmentation_layers()

        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.seg2map.warning_box)

    @debug_view.capture(clear_output=True)
    def load_feature_from_file(self, btn):
        # Prompt user to select a geojson file
        def file_chooser_callback(filechooser: FileChooser) -> None:
            try:
                if filechooser.selected:
                    print(
                        f"Loading ROIs from file: {os.path.abspath(filechooser.selected)}"
                    )
                    self.seg2map.load_feature_on_map(
                        file=os.path.abspath(filechooser.selected)
                    )
            except Exception as error:
                # renders error message as a box on map
                exception_handler.handle_exception(error, self.seg2map.warning_box)

        # create instance of chooser that calls callsfile_chooser_callback
        file_chooser = create_file_chooser(
            file_chooser_callback, title="Select a geojson file"
        )
        # clear row and close all widgets in self.file_chooser_row before adding new file_chooser
        self.clear_row(self.file_chooser_row)
        # add instance of file_chooser to row 4
        self.file_chooser_row.children = [file_chooser]

    @debug_view.capture(clear_output=True)
    def save_to_file_btn_clicked(self, btn):
        UI.debug_view.clear_output(wait=True)
        try:
            self.seg2map.save_feature_to_file(self.seg2map.rois)
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.seg2map.warning_box)

    @debug_view.capture(clear_output=True)
    def remove_all_from_map(self, btn):
        try:
            self.seg2map.remove_all()
        except Exception as error:
            # renders error message as a box on map
            exception_handler.handle_exception(error, self.seg2map.warning_box)

    def clear_debug_view(self, btn):
        UI.debug_view.clear_output()

    def clear_download_view(self, btn):
        UI.download_view.clear_output()
