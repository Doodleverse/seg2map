# standard python imports
import os
import glob
import logging

# internal python imports
from coastseg import common
from coastseg import zoo_model

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
    model_view = Output(layout={"border": "1px solid black"})
    run_model_view = Output(layout={"border": "1px solid black"})
    # FloodNet
    # uav_RGB_10class_7566797: https://zenodo.org/record/7566797
    # uav_RGB_10class_7566810: https://zenodo.org/record/7566810
    # CoastTrain
    # ortho_RGB_8class_7574784: https://zenodo.org/record/7574784
    # naip_RGB_5class_7566992: https://zenodo.org/record/7566992
    # naip_RGB_8class_7570583: https://zenodo.org/record/7570583
    # CHESAPEAKE
    # ortho_RGB_7class_7576904:  https://zenodo.org/record/7576904
    # OPEN EARTH NET
    # ortho_RGB_9class_7576894: https://zenodo.org/record/7576894
    # DeepGlobe
    # ortho_RGB_7clas_7576898 https://zenodo.org/record/7576898
    # EnviroAtlas
    # ortho_RGB_6class_7576909 https://zenodo.org/record/7576909
    # AAAI-Buildings
    # ortho_RGB_7class_7607895 https://zenodo.org/record/7607895

    def __init__(self):
        # Controls size of ROIs generated on map
        self.model_dict = {
            "sample_direc": None,
            "use_GPU": "0",
            "implementation": "ENSEMBLE",
            "model_type": "FloodNet_RGB_10class_7566797",
            "otsu": False,
            "tta": False,
        }
        # list of RGB  models available
        self.generic_landcover_models = [
            "Chesappeake_RGB_7clas_7576904",
            "OpenEarthNet_RGB_9class_7576894",
            "DeepGlobe_RGB_7clas_7576898",
            "EnviroAtlas_RGB_6class_7576909",
            "AAAI-Buildings_RGB_2class_7607895",
        ]
        # list of RGB  models available
        self.coastal_landcover_models = [
            "FloodNet_RGB_10class_7566797",
            "FloodNet_RGB_10class_7566810",
            "CoastTrain_RGB_8class_7574784",
            "CoastTrain_RGB_5class_7566992",
            "CoastTrain_RGB_8class_7570583",
            "Chesappeake_RGB_7clas_7576904",
            "AAAI-Buildings_RGB_2class_7607895",
        ]

        # Declare widgets and on click callbacks
        self._create_HTML_widgets()
        self._create_widgets()
        self._create_buttons()

    def create_dashboard(self):
        model_choices_box = HBox(
            [self.model_type_dropdown, self.model_dropdown, self.model_implementation]
        )
        checkboxes = HBox([self.GPU_checkbox, self.otsu_radio, self.tta_radio])
        instr_vbox = VBox(
            [
                self.instr_header,
                self.line_widget,
                self.instr_select_images,
                self.instr_run_model,
            ]
        )
        self.file_row = HBox([])
        self.warning_row = HBox([])
        display(
            checkboxes,
            model_choices_box,
            instr_vbox,
            self.use_select_images_button,
            self.line_widget,
            self.warning_row,
            self.file_row,
            UI_Models.model_view,
            self.run_model_button,
            UI_Models.run_model_view,
            self.open_results_button,
        )

    def _create_widgets(self):
        self.model_implementation = RadioButtons(
            options=["ENSEMBLE", "BEST"],
            value="ENSEMBLE",
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
            description="Model Input:",
            disabled=False,
        )
        self.model_type_dropdown.observe(self.handle_model_type_change, names="value")

        self.model_dropdown = ipywidgets.RadioButtons(
            options=self.generic_landcover_models,
            value=self.generic_landcover_models[0],
            description="Select Model:",
            disabled=False,
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

        self.instr_header = HTML(
            value="<h4>Click ONE of the following buttons:</h4>",
            layout=Layout(margin="0px 0px 0px 0px"),
        )

        self.instr_select_images = HTML(
            value='<b>1. Select Images Button</b> \
                <br> - This will open a pop up window where the RGB folder must be selected.<br>\
            - <span style="background-color:yellow;color: black;">WARNING :</span> You will not be able to see the files within the folder you select.<br>\
            ',
            layout=Layout(margin="0px 0px 0px 20px"),
        )

        self.instr_run_model = HTML(
            value='<b>2. Run Model Button</b> \
                <br> - Make sure to click Select Images Button<br>\
            - <span style="background-color:yellow;color: black;">WARNING :</span> You should not run multiple models on the same folder. Otherwise not all the model outputs\
            will be saved to the folder.<br>\
            ',
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
        if self.model_dict["sample_direc"] is None:
            self.launch_error_box(
                "Cannot Run Model",
                "You must click 'Use Data Directory' or 'Select Images' First",
            )
            return
        else:
            # gets GPU or CPU depending on whether use_GPU is True
            zoo_model.get_GPU(self.model_dict["use_GPU"])
            # Disable run and open results buttons while the model is running
            self.open_results_button.disabled = True
            self.run_model_button.disabled = True
            model_implementation = self.model_dict["implementation"]
            zoo_model_instance = zoo_model.Zoo_Model()
            logger.info(f"\nSelected directory: {self.model_dict['sample_direc']}\n")
            # get path to RGB directory for models
            # all other necessary files are relative to RGB directory
            RGB_path = common.get_RGB_in_path(self.model_dict["sample_direc"])
            self.model_dict["sample_direc"] = RGB_path
            print(f"Selected directory: {self.model_dict['sample_direc']}")

            # specify dataset_id to download selected model
            dataset_id = self.model_dict["model_type"]
            use_otsu = self.model_dict["otsu"]
            use_tta = self.model_dict["tta"]
            # First download the specified model
            zoo_model_instance.download_model(model_implementation, dataset_id)
            # Get weights as list
            weights_list = zoo_model_instance.get_weights_list(model_implementation)
            # Load the model from the config files
            model, model_list, config_files, model_types = zoo_model_instance.get_model(
                weights_list
            )
            metadatadict = zoo_model_instance.get_metadatadict(
                weights_list, config_files, model_types
            )
            # # Compute the segmentation
            zoo_model_instance.compute_segmentation(
                self.model_dict["sample_direc"],
                model_list,
                metadatadict,
                use_tta,
                use_otsu,
            )
            # Enable run and open results buttons when model has executed
            self.run_model_button.disabled = False
            self.open_results_button.disabled = False

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
            sample_direc = os.path.abspath(filechooser.selected)
            print(f"The images in the folder will be segmented :\n{sample_direc} ")
            jpgs = glob.glob1(sample_direc + os.sep, "*jpg")
            if jpgs == []:
                self.launch_error_box(
                    "File Not Found",
                    "The directory contains no jpgs! Please select a directory with jpgs.",
                )
            elif jpgs != []:
                self.model_dict["sample_direc"] = sample_direc

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
