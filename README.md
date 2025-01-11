# Accomplish Calisthenics Skills Classification through Raw and Depth Patches

This repository contains the codes of the work "Accomplish Calisthenics Skills Classification through Raw and Depth Patches" create for classifying calisthenics skills using depth and raw image patches.


## Installation

1.  **Create conda environment:**

    ```bash
    conda create -n calisthenics_env python=3.12
    conda activate calisthenics_env  
    ```

2.  **Clone repositories:**
    Since it leverages the Depth Anything v2 and YOLOv10 libraries for efficient depth estimation and person detection. You need to clone this repository in `src/`

    ```bash
    cd src
    git clone [https://github.com/DepthAnything/Depth-Anything-V2.git](https://github.com/DepthAnything/Depth-Anything-V2.git) depth_anything_v2
    git clone [https://github.com/THU-MIG/yolov10.git](https://github.com/THU-MIG/yolov10.git) yolov10
    cd ..
    ```

3.  **Link YOLOv10 files (modify paths if needed):**
    In the `src/yolov10/` directory, there are two files: `bbox_operations.py` and `image_inference.py` that need to be copied into the main directory of YOLOv10 repository.
    ```bash
    cp -r src/yolo_files/* yolov10/
    ```

4.  **Install dependencies:**
    Run the following command to install all the dependecies needed for the project.
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure
    After cloning the repositories and installing the dependencies, the project structure should look like this:
*   `Depth-Anything-V2`: Linked Depth Anything v2 repository.
*   `yolov10`: Linked YOLOv10 repository.
*   `src`: Contains the code for inference, training, and evaluation.
*   `data`: Directory for input images for inference.
*   `requirements.txt`: Lists Python dependencies.

## How to Use

1.  **Prepare your dataset:**

    *   Place your depth images, raw image patches, and corresponding class labels in the `data` directory. Ensure proper organization and subfolders for different categories if needed. Refer to Depth Anything v2 and YOLOv10 documentation for specific data format requirements.

2.  **Train your model (modify script according to your project):**

    *   Edit the script in `src` for your CNN architecture, training parameters, and data loading logic.
    *   Run training: `python train.py` (replace `train.py` with your script name)

3.  **Evaluate your model (modify script according to your project):**

    *   Edit the script in `src` for evaluation metrics and data loading logic.
    *   Run evaluation: `python evaluate.py` (replace `evaluate.py` with your script name)

4.  **Perform inference:**

    *   Edit the script in `inference` to specify the input image and output format.
    *   Run inference: `python inference.py image.jpg` (replace `image.jpg` with your input image path)
