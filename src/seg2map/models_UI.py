# standard python imports
import os
import logging

# internal python imports
from seg2map import common
from seg2map import zoo_model

# external python imports
import ipywidgets
from IPython.display import display
from ipywidgets import Button
from ipywidgets import ToggleButton
from ipywidgets import HBox
from ipywidgets import VBox
from ipywidgets import Layout
from ipywidgets import HTML
from ipywidgets import RadioButtons
from ipywidgets import Output
from ipyfilechooser import FileChooser

# icons sourced from https://fontawesome.com/v4/icons/

logger = logging.getLogger(__name__)


def create_dir_chooser(callback, title: str = None):
    padding = "0px 0px 0px 5px"  # upper, right, bottom, left
    data_path = os.path.join(os.getcwd(), "data")
    if os.path.exists(data_path):
        data_path = os.path.join(os.getcwd(), "data")
    else:
        data_path = os.getcwd()
    # creates a unique instance of filechooser and button to close filechooser
    dir_chooser = FileChooser(data_path)
    dir_chooser.dir_icon = os.sep
    # Switch to folder-only mode
    dir_chooser.show_only_dirs = True
    if title is not None:
        dir_chooser.title = f"<b>{title}</b>"
    dir_chooser.register_callback(callback)

    close_button = ToggleButton(
        value=False,
        tooltip="Close Directory Chooser",
        icon="times",
        button_style="primary",
        layout=Layout(height="28px", width="28px", padding=padding),
    )

    def close_click(change):
        if change["new"]:
            dir_chooser.close()
            close_button.close()

    close_button.observe(close_click, "value")
    chooser = HBox([dir_chooser, close_button])
    return chooser


class UI_Models:
    # all instances of UI will share the same debug_view
    model_view = Output()
    run_model_view = Output()

    def __init__(self):
        # Controls size of ROIs generated on map
        self.model_dict = {
            "sample_direc": None,
            "use_GPU": "0",
            "implementation": "BEST",
            "model_type": "OpenEarthNet_RGB_9class_7576894",
            "otsu": False,
            "tta": False,
        }
        # list of RGB  models available
        self.generic_landcover_models = [
            "OpenEarthNet_RGB_9class_7576894",
            "DeepGlobe_RGB_7class_7576898",
            "EnviroAtlas_RGB_6class_7576909",
            "AAAI-Buildings_RGB_2class_7607895",
            "aaai_floodedbuildings_RGB_2class_7622733",
            "xbd_building_RGB_2class_7613212",
            "xbd_damagedbuilding_RGB_4class_7613175",
        ]

        # list of RGB  models available
        self.coastal_landcover_models = [
            "chesapeake_RGB_7class_7576904",
            "orthoCT_RGB_2class_7574784",
            "orthoCT_RGB_5class_7566992",
            "orthoCT_RGB_5class_segformer_7641708",
            "orthoCT_RGB_8class_7570583",
            "orthoCT_RGB_8class_segformer_7641724",
            "chesapeake_7class_segformer_7677506",
        ]  # @todo add barrier islands model when its up

        self.session_name = ""
        self.inputs_directory = ""
        # Declare widgets and on click callbacks
        self._create_HTML_widgets()
        self._create_widgets()
        self._create_buttons()

    def set_inputs_directory(self, full_path: str):
        self.inputs_directory = os.path.abspath(full_path)

    def get_inputs_directory(
        self,
    ):
        return self.inputs_directory

    def set_session_name(self, name: str):
        self.session_name = str(name).strip()

    def get_session_name(
        self,
    ):
        return self.session_name

    def get_session_selection(self):
        output = Output()
        self.session_name_text = ipywidgets.Text(
            value="",
            placeholder="Enter a session name",
            description="Session Name:",
            disabled=False,
            style={"description_width": "initial"},
        )

        enter_button = ipywidgets.Button(description="Enter")

        @output.capture(clear_output=True)
        def enter_clicked(btn):
            session_name = str(self.session_name_text.value).strip()
            session_path = common.create_directory(os.getcwd(), "sessions")
            new_session_path = os.path.join(session_path, session_name)
            if os.path.exists(new_session_path):
                print(f"Session {session_name} already exists at {new_session_path}")
            elif not os.path.exists(new_session_path):
                print(f"Session {session_name} will be created at {new_session_path}")
                self.set_session_name(session_name)

        enter_button.on_click(enter_clicked)
        session_name_controls = HBox([self.session_name_text, enter_button])
        return VBox([session_name_controls, output])

    def create_dashboard(self):
        model_choices_box = HBox(
            [self.model_type_dropdown, self.model_dropdown, self.model_implementation]
        )
        checkboxes = HBox([self.otsu_radio, self.tta_radio])
        instr_vbox = VBox(
            [
                self.instr_select_images,
                self.instr_run_model,
            ]
        )
        self.file_row = HBox([])
        self.warning_row = HBox([])
        display(
            checkboxes,
            model_choices_box,
            self.get_session_selection(),
            instr_vbox,
            self.use_select_images_button,
            self.warning_row,
            self.file_row,
            UI_Models.model_view,
            self.run_model_button,
            UI_Models.run_model_view,
        )

    def _create_widgets(self):
        self.model_implementation = RadioButtons(
            options=["BEST", "ENSEMBLE"],
            value="BEST",
            description="Select:",
            disabled=False,
        )
        self.model_implementation.observe(self.handle_model_implementation, "value")

        self.otsu_radio = RadioButtons(
            options=["Enabled", "Disabled"],
            value="Disabled",
            description="Otsu Threshold:",
            disabled=False,
            style={"description_width": "initial"},
        )
        self.otsu_radio.observe(self.handle_otsu, "value")

        self.tta_radio = RadioButtons(
            options=["Enabled", "Disabled"],
            value="Disabled",
            description="Test Time Augmentation:",
            disabled=False,
            style={"description_width": "initial"},
        )
        self.tta_radio.observe(self.handle_tta, "value")

        self.model_type_dropdown = ipywidgets.RadioButtons(
            options=["Generic Landcover", "Coastal Landcover"],
            value="Generic Landcover",
            description="Model Type:",
            disabled=False,
            style={"description_width": "initial"},
        )
        self.model_type_dropdown.observe(self.handle_model_type_change, names="value")

        self.model_dropdown = ipywidgets.Dropdown(
            options=self.generic_landcover_models,
            value=self.generic_landcover_models[0],
            description="Select Model:",
            disabled=False,
            style={"description_width": "initial"},
        )
        self.model_dropdown.observe(self.handle_model_dropdown, "value")

        # Allow user to enable GPU
        self.GPU_checkbox = ipywidgets.widgets.Checkbox(
            value=False, description="Use GPU", disabled=False, indent=False
        )
        self.GPU_checkbox.observe(self.handle_GPU_checkbox, "value")

    def _create_buttons(self):
        # button styles
        load_style = dict(button_color="#69add1")
        action_style = dict(button_color="#ae3cf0")

        self.run_model_button = Button(
            description="Run Model",
            style=action_style,
            icon="fa-bolt",
        )
        self.run_model_button.on_click(self.run_model_button_clicked)

        self.use_select_images_button = Button(
            description="Select Images",
            style=load_style,
            icon="fa-file-image-o",
        )
        self.use_select_images_button.on_click(self.use_select_images_button_clicked)
        self.open_results_button = Button(
            description="Open Results",
            style=load_style,
            icon="folder-open-o",
        )
        self.open_results_button.on_click(self.open_results_button_clicked)

    def _create_HTML_widgets(self):
        """create HTML widgets that display the instructions.
        widgets created: instr_create_ro, instr_save_roi, instr_load_btns
         instr_download_roi
        """
        self.line_widget = HTML(
            value="____________________________________________________"
        )

        self.instr_select_images = HTML(
            value="<b>1. Select Images</b> \
                <br> - Select an ROI directory or a directory containing at least one ROI subdirectory.\nExample: ./data/dataset1/ID_e7CxBi_dates_2010-01-01_to_2014-12-31",
            layout=Layout(margin="0px 0px 0px 20px"),
        )

        self.instr_run_model = HTML(
            value="<b>2. Run Model </b> \
                <br> - Click Select Images first, then click run model",
            layout=Layout(margin="0px 0px 0px 20px"),
        )

    def handle_model_implementation(self, change):
        self.model_dict["implementation"] = change["new"]

    def handle_model_dropdown(self, change):
        # 2 class model has not been selected disable otsu threhold
        if "2class" not in change["new"]:
            if self.otsu_radio.value == "Enabled":
                self.model_dict["otsu"] = False
                self.otsu_radio.value = "Disabled"
            self.otsu_radio.disabled = True
        # 2 class model was selected enable otsu threhold radio button
        if "2class" in change["new"]:
            self.otsu_radio.disabled = False

        logger.info(f"change: {change}")
        self.model_dict["model_type"] = change["new"]

    def handle_GPU_checkbox(self, change):
        if change["new"] == True:
            self.model_dict["use_GPU"] = "1"
        elif change["new"] == False:
            self.model_dict["use_GPU"] = "0"

    def handle_otsu(self, change):
        if change["new"] == "Enabled":
            self.model_dict["otsu"] = True
        if change["new"] == "Disabled":
            self.model_dict["otsu"] = False

    def handle_tta(self, change):
        if change["new"] == "Enabled":
            self.model_dict["tta"] = True
        if change["new"] == "Disabled":
            self.model_dict["tta"] = False

    def handle_model_type_change(self, change):
        if change["new"] == "Generic Landcover":
            self.model_dropdown.options = self.generic_landcover_models
        if change["new"] == "Coastal Landcover":
            self.model_dropdown.options = self.coastal_landcover_models

    @run_model_view.capture(clear_output=True)
    def run_model_button_clicked(self, button):
        session_name = self.get_session_name()
        inputs_directory = self.get_inputs_directory()
        if session_name == "":
            self.launch_error_box(
                "Cannot Run Model",
                "Must enter a session name first",
            )
            return
        if inputs_directory == "":
            self.launch_error_box(
                "Cannot Run Model",
                "Must click 'Select Images' first",
            )
            return
        if not common.check_id_subdirectories_exist(inputs_directory):
            self.launch_error_box(
                "Cannot Run Model",
                "You must select a directory that contains ROI subdirectories. Example ROI name: 'ID_e7CxBi_dates_2010-01-01_to_2014-12-31'",
            )
            return
        # Disable run and open results buttons while the model is running
        self.run_model_button.disabled = True

        # gets GPU or CPU depending on whether use_GPU is True
        use_GPU = self.model_dict["use_GPU"]
        model_implementation = self.model_dict["implementation"]
        model_id = self.model_dict["model_type"]
        use_otsu = self.model_dict["otsu"]
        use_tta = self.model_dict["tta"]

        zoo_model_instance = zoo_model.ZooModel()
        try:
            zoo_model_instance.run_model(
                model_implementation,
                session_name=session_name,
                src_directory=inputs_directory,
                model_id=model_id,
                use_GPU=use_GPU,
                use_otsu=use_otsu,
                use_tta=use_tta,
        )
        finally:
            # Enable run and open results buttons when model has executed
            self.run_model_button.disabled = False

    @run_model_view.capture(clear_output=True)
    def open_results_button_clicked(self, button):
        """open_results_button_clicked on click handler for 'open results' button.

        prints location of model outputs

        Args:
            button (Button): button that was clicked

        Raises:
            FileNotFoundError: raised when the directory where the model outputs are saved does not exist
        """
        if self.model_dict["sample_direc"] is None:
            self.launch_error_box(
                "Cannot Open Results", "You must click 'Run Model' first"
            )
        else:
            # path to directory containing model outputs
            model_results_path = os.path.abspath(self.model_dict["sample_direc"])
            if not os.path.exists(model_results_path):
                self.launch_error_box(
                    "File Not Found",
                    "The directory for the model outputs could not be found",
                )
                raise FileNotFoundError
            else:
                print(f"Model outputs located at:\n{model_results_path}")

    @model_view.capture(clear_output=True)
    def load_callback(self, filechooser: FileChooser) -> None:
        if filechooser.selected:
            inputs_directory = os.path.abspath(filechooser.selected)
            self.set_inputs_directory(inputs_directory)
            # for root, dirs, files in os.walk(inputs_directory):
            #     # if any directory contains jpgs then set inputs directory to selected directory
            #     jpgs = glob.glob(os.path.join(root, "*jpg"))
            #     if len(jpgs) > 0:
            #         self.set_inputs_directory(inputs_directory)
            #         return
            # self.launch_error_box(
            #     "File Not Found",
            #     "The directory contains no jpgs! Please select a directory with jpgs.",
            # )

    @model_view.capture(clear_output=True)
    def use_select_images_button_clicked(self, button):
        # Prompt the user to select a directory of images
        file_chooser = create_dir_chooser(
            self.load_callback, title="Select directory of images"
        )
        # clear row and close all widgets in self.file_row before adding new file_chooser
        common.clear_row(self.file_row)
        # add instance of file_chooser to self.file_row
        self.file_row.children = [file_chooser]

    def launch_error_box(self, title: str = None, msg: str = None):
        # Show user error message
        warning_box = common.create_warning_box(title=title, msg=msg)
        # clear row and close all widgets in self.file_row before adding new warning_box
        common.clear_row(self.warning_row)
        # add instance of warning_box to self.warning_row
        self.warning_row.children = [warning_box]
