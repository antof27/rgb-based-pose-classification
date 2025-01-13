import os
import subprocess
import sys
import concurrent.futures
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import efficientnet_v2_l
import time

# Directories for input and output
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10  
input_dir = '/home/afinocchiaro/dm/image_path/raw_i'
output_dir = '/home/afinocchiaro/dm/output_pipeline'
depth_script_dir = '/home/afinocchiaro/dm/src/Depth-Anything-V2'
yolo_script_dir = '/home/afinocchiaro/mr/src/yolov10'

trained_weights_path = '/home/afinocchiaro/mr/src/yolov10/last.pt'
os.makedirs(output_dir, exist_ok=True)

# Parameters for depth processing
encoder = 'vitl'
pred_only = True
MODE = 'depth'

if len(sys.argv) > 1:
    MODE = sys.argv[1]


overall_start_time = time.time()
def timed_execution(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    print(f"{func.__name__} completed in {elapsed_time:.2f} seconds.")
    return result


# Command to run the depth processing script
depth_command = [
    "python3", "run.py",
    "--encoder", encoder,
    "--img-path", input_dir,
    "--outdir", output_dir,
]

if pred_only:
    depth_command.append("--pred-only")


# Class name mapping
class_names = ['bl', 'fl', 'flag', 'ic', 'mal', 'none', 'oafl', 'oahs', 'pl', 'vsit']

def initialize_model_efficientnet(num_classes, weights_path=None, activation=torch.nn.Softplus()):
    """
    Initializes the EfficientNet V2-L model, modifies the classifier for custom class outputs, and optionally loads weights.
    Args:
        num_classes (int): Number of output classes for the classification task.
        weights_path (str, optional): Path to the model weights.
        activation (torch.nn.Module): Activation function to be used after the classifier.
    Returns:
        model (torch.nn.Module): The modified EfficientNet model.
    """
    # Load the EfficientNet model without pretrained weights
    model = efficientnet_v2_l(weights=None)  # Start with no weights
    
    # Modify the classifier to match the number of classes
    model.classifier[1] = torch.nn.Sequential(
        torch.nn.Linear(model.classifier[1].in_features, num_classes),
        activation
    )
    
    # If weights_path is provided, load the model weights
    if weights_path:
        checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f"Loaded weights from {weights_path}")
    
    return model



def get_transform():
    """
    Defines the image transformation pipeline including resizing and tensor conversion.
    Returns:
        transform (torchvision.transforms.Compose): The composed transformation pipeline.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),  
    ])

# Load and process images from a folder
def load_images_from_folder(folder, transform):
    """
    Loads images from a specified folder and applies transformations.
    Args:
        folder (str): Path to the folder containing the images.
        transform (torchvision.transforms.Compose): The transformation to apply to each image.
    Returns:
        images (list): List of image tensors.
        filenames (list): List of filenames corresponding to the loaded images.
    """
    images = []
    filenames = []
    
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'): 
            file_path = os.path.join(folder, filename)
            image = Image.open(file_path).convert("RGB")  
            image_tensor = transform(image)  
            images.append(image_tensor)
            filenames.append(filename)
    
    return images, filenames

# Function to run depth processing
def run_depth_processing():
    try:
        subprocess.run(depth_command, check=True, cwd=depth_script_dir)
        print(f"Depth processing completed. Output saved in {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error during depth processing: {e}")

# Function to run YOLO inference
def run_yolo_inference():
    try:
        sys.path.append(yolo_script_dir)
        from image_inference import run_yolo_inference
        
        predictions = run_yolo_inference(trained_weights_path, input_dir, output_dir)
        print(f"YOLO inference completed. Predictions saved in {output_dir}")
        return predictions
    except Exception as e:
        print(f"Error during YOLO inference: {e}")
        return None

# Function to process and crop images after depth processing
def process_images_after_depth(predictions):
    try:
        sys.path.append(yolo_script_dir)
        from bbox_operations import process_and_crop_images
        
        if predictions:
            if MODE == "depth":
                process_and_crop_images(predictions, output_dir, output_dir, MODE)
            else:
                process_and_crop_images(predictions, input_dir, output_dir, MODE)
                
            print(f"Image processing completed. Cropped images saved in {output_dir}")
        else:
            print("No predictions to process.")
    except Exception as e:
        print(f"Error during image processing: {e}")

# Function to process a single image for EfficientNet inference
def process_single_image(image_path, transform, device):
    image = Image.open(image_path).convert("RGB")  
    image_tensor = transform(image).unsqueeze(0).to(device)  
    return image_tensor

# Function to perform EfficientNet inference
def perform_inference(image_folder, model, device):
    transform = get_transform()

    images = []
    filenames = []
    print(f"Loading images from {image_folder}...")

    # Load and process all images in the folder
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg'):
            file_path = os.path.join(image_folder, filename)
            image_tensor = process_single_image(file_path, transform, device)  # Process one image
            images.append(image_tensor)
            filenames.append(filename)
    
    if not images:
        print("No images found in the folder.")
        return []

    dataset = torch.cat(images, dim=0).to(device)  
    
    results = []
    print("Performing inference...")
    model.eval()
    with torch.no_grad():
        outputs = model(dataset)  
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()  
        
        # Convert predicted class IDs to class names
        for i, pred in enumerate(predictions):
            # print(f"Predicted class: {pred}")
            # print(f"Filename: {filenames[i]}")
            results.append((filenames[i], class_names[pred]))
    
    return results

# Function to run EfficientNet inference
def run_efficientnet_inference():
    try:
        print("Running EfficientNet inference...")
        
        # Specify the path to the saved weights
        weights_path = f'/path/to/efficientnet/weights/{MODE}.pth'
        
        # Initialize model with weights
        model = initialize_model_efficientnet(NUM_CLASSES, weights_path=weights_path).to(DEVICE)
        
        results = perform_inference(output_dir, model, DEVICE)
        print("EfficientNet inference completed.")
        for filename, pred_class in results:
            print(f"Image: {filename}, Predicted Class: {pred_class}")
    except Exception as e:
        print(f"Error during EfficientNet inference: {e}")


# Main execution with concurrent processing for depth and YOLO
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Submit task for depth processing
    #Comment to not compute depth
    depth_future = executor.submit(timed_execution, run_depth_processing)

    # # Run YOLO inference independently
    #Comment to not compute patches
    yolo_future = executor.submit(timed_execution, run_yolo_inference)

    # # Wait for depth processing to complete
    #Comment to not compute depth
    depth_future.result()  # Blocks until depth processing is done

    # # Get predictions from YOLO inference
    #Comment to not compute patches
    predictions = yolo_future.result()

    # # Process images after depth processing completes
    #Comment to not compute depth patches
    timed_execution(process_images_after_depth, predictions)

    # Perform EfficientNet inference
    timed_execution(run_efficientnet_inference)

# Calculate overall time
overall_elapsed = time.time() - overall_start_time
print(f"Overall execution completed in {overall_elapsed:.2f} seconds.")
