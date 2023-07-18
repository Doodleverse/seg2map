# Testing & Debugging Script for Seg2Map
# This script is designed for testing and debugging the Seg2Map application. It runs multiple models on a provided input directory containing RGB images,
# using the specified implementation and model types.The results are logged and saved for analysis and evaluation.
# The script allows for command-line arguments to specify the input directory path.
#
# # Author: Sharon Fitzpatrick
# Date: 7/18/2023
#
# To run this script, follow these steps:
# Make sure you have Python installed and the necessary dependencies for the script.
# 1. Open a command prompt or terminal.
# 2. Replace <your path here> in the command below with the path to the ROI's RGB directory:
#    python test_models.py -P "<your path here>"" -I "BEST"
# 2. Execute the command. For example, if the RGB directory is located at C:\development\doodleverse\seg2map\seg2map\data\new_data, the command would be:
#    python test_models.py -P "C:\development\doodleverse\seg2map\seg2map\data\new_data" -I "BEST"


import argparse
from seg2map import log_maker
from seg2map.zoo_model import ZooModel

# from transformers import TFSegformerForSemanticSegmentation
# import tensorflow as tf

# alternatively you can hard code your own variables
# INPUT_DIRECTORY = r"C:\development\doodleverse\seg2map\seg2map\data\new_data"
INPUT_DIRECTORY = r"C:\development\doodleverse\seg2map\seg2map\data\download_group1"

IMPLEMENTATION = "BEST"  # "ENSEMBLE" or "BEST"


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run models on provided input directory."
    )
    parser.add_argument(
        "-P",
        "--path",
        type=str,
        help="Path to an ROI's RGB directory from the data directory",
    )
    parser.add_argument(
        "-I",
        "--implementation",
        type=str,
        help="BEST or ENSEMBLE",
    )
    return parser.parse_args()


def print_model_info(model_selected, session_name, input_directory):
    """Print information about the selected model."""
    print(f"Running model {model_selected}")
    print(f"session_name: {session_name}")
    print(f"model_selected: {model_selected}")
    print(f"sample_directory: {input_directory}")


def run_model(model_dict):
    """Run the Seg2Map model with given parameters."""
    zoo_model_instance = ZooModel()
    zoo_model_instance.run_model(
        model_dict["implementation"],
        model_dict["session_name"],
        model_dict["sample_direc"],
        model_id=model_dict["model_type"],
        use_GPU="0",
        use_otsu=model_dict["otsu"],
        use_tta=model_dict["tta"],
    )


def main():
    args = parse_arguments()

    # Get input directory and implementation from command-line arguments or use default values
    input_directory = args.path or INPUT_DIRECTORY
    implementation = args.implementation or IMPLEMENTATION

    print(f"Using input_directory: {input_directory}")
    print(f"Using implementation: {implementation}")

    # List of models that will be tested
    available_models = [
        "OpenEarthNet_RGB_9class_7576894",
        "DeepGlobe_RGB_7class_7576898",
        "EnviroAtlas_RGB_6class_7576909",
        "AAAI-Buildings_RGB_2class_7607895",
        "aaai_floodedbuildings_RGB_2class_7622733",
        "xbd_building_RGB_2class_7613212",
        "xbd_damagedbuilding_RGB_4class_7613175",
        "chesapeake_RGB_7class_7576904",
        "orthoCT_RGB_2class_7574784",
        "orthoCT_RGB_5class_7566992",
        "orthoCT_RGB_5class_segformer_7641708",
        "orthoCT_RGB_8class_7570583",
        "orthoCT_RGB_8class_segformer_7641724",
        "chesapeake_7class_segformer_7677506",
    ]

    for model_selected in available_models:
        session_name = model_selected + "_" + implementation + "_" + "session"
        print_model_info(model_selected, session_name, input_directory)

        # Load the basic zoo_model settings
        model_dict = {
            "sample_direc": input_directory,
            "session_name": session_name,
            "use_GPU": "0",
            "implementation": implementation,
            "model_type": model_selected,
            "otsu": False,
            "tta": False,
        }
        run_model(model_dict)


if __name__ == "__main__":
    main()
