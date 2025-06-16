import os
import subprocess
import sys
import concurrent.futures
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import efficientnet_v2_l
import time


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10  

input_dir = '/path/to/input_dir'
output_dir = '/path/to/outpu_dir'

depth_script_dir = 'path/to/dav2'
yolo_script_dir = '/path/to/yolo_scripts'

yolo_weights_path = '/path/to/yolo/checkpoint'
efficient_weights_path = '/path/to/efficient/checkpoint'

os.makedirs(output_dir, exist_ok=True)

# Parameters for depth processing
encoder = 'vits'
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

    model = efficientnet_v2_l(weights=None)  
    
    model.classifier[1] = torch.nn.Sequential(
        torch.nn.Linear(model.classifier[1].in_features, num_classes),
        activation
    )
    
    if weights_path:
        checkpoint = torch.load(efficient_weights_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint)
    
    return model



def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),  
    ])

# Load and process images from a folder, applying the necessary transformations
def load_images_from_folder(folder, transform):
    
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
        
        predictions = run_yolo_inference(yolo_weights_path, input_dir, output_dir)
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
            image_tensor = process_single_image(file_path, transform, device)  
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
        
        for i, pred in enumerate(predictions):
            results.append((filenames[i], class_names[pred]))
    
    return results

# Function to run EfficientNet inference
def run_efficientnet_inference():
    try:
        print("Running EfficientNet inference...")
        
        # Initialize model with weights
        model = initialize_model_efficientnet(NUM_CLASSES, weights_path=efficient_weights_path).to(DEVICE)
        
        results = perform_inference(output_dir, model, DEVICE)
        print("EfficientNet inference completed.")
        for filename, pred_class in results:
            print(f"Image: {filename}, Predicted Class: {pred_class}")
    except Exception as e:
        print(f"Error during EfficientNet inference: {e}")


# Main execution with concurrent processing for depth and YOLO
with concurrent.futures.ThreadPoolExecutor() as executor:
    
    if MODE == "depth":
        
        yolo_future = executor.submit(timed_execution, run_yolo_inference)
        predictions = yolo_future.result()
        timed_execution(process_images_after_depth, predictions)
        depth_future = executor.submit(timed_execution, run_depth_processing)
        depth_future.result()

    # # Run YOLO inference independently
    else:
        yolo_future = executor.submit(timed_execution, run_yolo_inference)
        predictions = yolo_future.result()


    # Perform EfficientNet inference
    timed_execution(run_efficientnet_inference)

# Calculate overall time
overall_elapsed = time.time() - overall_start_time
print(f"Overall execution completed in {overall_elapsed:.2f} seconds.")
