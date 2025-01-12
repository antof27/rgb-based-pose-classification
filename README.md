# Accomplish Calisthenics Skills Classification through Raw and Depth Patches

This repository contains the codes of the work "Accomplish Calisthenics Skills Classification through Raw and Depth Patches" created for classifying calisthenics skills using raw and depth image patches.

## Installation

1.  **Create conda environment and clone this repository:**

    ```bash
    conda create -n calisthenics_env python=3.12
    conda activate calisthenics_env
    git clone https://github.com/antof27/rgb-based-pose-classification.git
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
    Run the following command to install all the dependencies needed for the project.
    ```bash
    pip install -r requirements.txt

4.  **Copy yolov10 files:**
    In the `src/yolov10/` directory, there are two files: `bbox_operations.py` and `image_inference.py` that need to be copied into the main directory of YOLOv10 repository.
    ```bash
    cp -r src/yolo_files/* yolov10/

    ```
5.  **Download YOLOv10 pre-trained weights:**

    For running inference with raw and depth patches, download the YOLOv10 pre-trained weights from [YOLOv10 weights](https://www.kaggle.com/code/cubeai/person-detection-with-yolov10/output).


7. **Download the CNN weights**

   In order to fine-tune or perform inference with the pre-trained weights, you can download the pre-trained weights from the following link: [EfficientNetv2 weights](https://www.dropbox.com/home/cnn_weights).
   


## Project Structure

After cloning the repositories and installing the dependencies, the project structure should look like this:

<pre>
src/
├── yolo_files/
├── yolov10/
├── Depth-Anything-V2/
├── inference/
└── training_eval/

data/
├── csv_files/
├── input_inference_images/
└── output_inference_images/

Finocchiaro_ACSCtRDP.pdf
    
requirements.txt
</pre>

## How to Use

1.  **To train and test on your own dataset**

    *   Place your depth images, raw image patches or depth patches in the `data` directory. You should include two `.csv` files with `filename` columnn and `label` column. See the examples in the `data/csv_files` directory.
        If you want to reproduce the same experiments, you can ask me to provide the image dataset.
    *   Modify paths and hyperparameters in `src/training_eval/main_script.py` and run the script specifying some parameters in the code or in the command, for example:
        ```bash
        python3 model.py --weights pretrained --mode normal --n_gpu 1
        ```
        where `--weights` can be [non-pretrained, pretrained], `--mode` refers to images and can be [normal, depth], `--n_gpu` refers to the GPU id (in case of multiple GPUs).
        

2.  **Perform inference:**
    *   Place your input images in the `data/input_inference_images` directory.
    *   Edit the script `full_inference.py` in `src/inference` to specify the input, output images and weights paths.
    *   Run inference: 
        ```bash
        python3 full_inference.py
        ```
