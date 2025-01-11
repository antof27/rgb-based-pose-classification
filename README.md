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
3.  **Install dependencies:**
    Run the following command to install all the dependecies needed for the project.
    ```bash
    pip install -r requirements.txt

4.  **Copy yolov10 files:**
    In the `src/yolov10/` directory, there are two files: `bbox_operations.py` and `image_inference.py` that need to be copied into the main directory of YOLOv10 repository.
    ```bash
    cp -r src/yolo_files/* yolov10/

    ```
5.  **Download YOLOv10 pre-trained weights:**

    For running inference with raw and depth patches, download the yolov10 pre-trained weights:

    ```bash
    wget [https://raw.githubusercontent.com/ultralytics/yolov10/master/weights/yolov10n.pt](https://raw.githubusercontent.com/ultralytics/yolov10/master/weights/yolov10n.pt) -P yolov10/
    # Or download from kaggle and place in yolov10/
    # [https://www.kaggle.com/code/cubeai/person-detection-with-yolov10/output](https://www.kaggle.com/code/cubeai/person-detection-with-yolov10/output)
    ```

## Project Structure

After cloning the repositories and installing the dependencies, the project structure should look like this:

*   `Depth-Anything-V2`: Linked Depth Anything v2 repository.
*   `yolov10`: Linked YOLOv10 repository.
*   `src`: Contains the code for inference, training, and evaluation.
*   `data`: Directory for input images for inference.
*   `requirements.txt`: Lists Python dependencies.

## How to Use

1.  **To train and test on your own dataset**

    *   Place your depth images, raw image patches or depth patches in the `data` directory. You should include two `.csv` files with `filename` columnn and `label` column. See the examples in the `data` directory.
    *   Modify paths in `src/training_eval/main_script.py` to point to your data.

2.  **Perform inference:**
    *  Place your input image in the `data` directory.
    *   Edit the script in `inference` to specify the input image and output format.
    *   Run inference: `python inference.py image.jpg` (replace `image.jpg` with your input image path)
