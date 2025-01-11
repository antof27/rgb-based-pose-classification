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
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git
    git clone https://github.com/THU-MIG/yolov10.git
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


├── src/                          # Contains code for inference, training, and evaluation
|    ├── yolo_files/                # Files to be copied to yolov10/ directory
|    ├── yolov10/          
|    ├── Depth-Anything-V2/        
│    ├── inference/                # Inference scripts
│    ├── training_eval/            # Training and evaluation scripts
│              └── cnn_weights/
│    
├── data/                         # Directory for input images (inference)
│    ├── csv_files/             
│    ├── input_inference_images/              
│    └── output_inference_images/                   
└── requirements.txt              # Python dependencies


## How to Use

1.  **To train and test on your own dataset**

    *   Place your depth images, raw image patches or depth patches in the `data` directory. You should include two `.csv` files with `filename` columnn and `label` column. See the examples in the `data/csv_files` directory.
        If you want to reproduce the same experiments, you can ask me to provide the image dataset.
    *   Modify paths in `src/training_eval/main_script.py` to point to your data.

2.  **Perform inference:**
    *   Place your input images in the `data/input_inference_images` directory.
    *   Edit the script `full_inference.py` in `src/inference` to specify the input, output images and weights paths.
    *   Run inference: 
        ```bash
        python3 full_inference.py
        ```
