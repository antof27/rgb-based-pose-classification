import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from torchvision import transforms
from torch.nn import Linear
import os
from PIL import Image

# Path to the model weights
WEIGHTS_PATH = "/home/afinocchiaro/dm/src/EfficientNetv2/pretrained/depth_patches/final_model_weights.pth"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the EfficientNet model
def initialize_model_efficientnet(num_classes, activation = nn.Softplus()):

    # Load the EfficientNet model with or without pretrained weights
    model = efficientnet_v2_l(weights=None)  # Start with no weights
    
    # Modify the classifier
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, num_classes),
        activation
    )
    return model

# Define the transformations to be applied to each image
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

# Load images from folder and ensure correct shape
def load_images_from_folder(folder, transform):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            file_path = os.path.join(folder, filename)
            image = Image.open(file_path).convert("RGB")
            image_tensor = transform(image)  # Apply transformations
            images.append(image_tensor)
            filenames.append(filename)
    return images, filenames

# Perform inference on images (without batching)
def perform_inference(image_folder, model, weights_path):
    transform = get_transform()
    print(f"Loading images from {image_folder}...")
    images, filenames = load_images_from_folder(image_folder, transform)

    if not images:
        print("No images found in the folder.")
        return []
    
    # Load model weights
    model.load_state_dict(torch.load(weights_path))
    model.to(DEVICE)  # Ensure model is on the right device
    
    # Prepare images in the correct format for EfficientNet
    dataset = torch.stack(images).to(DEVICE)  # Stack images into a batch
    model.eval()  # Set the model to evaluation mode

    results = []
    with torch.no_grad():
        outputs = model(dataset)  # Pass all images in the batch
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

        # Save results with corresponding filenames
        for i, pred in enumerate(predictions):
            results.append((filenames[i], pred))

    return results

# Example usage:
# model = initialize_model_efficientnet(num_classes=10, activation_name="ReLU")
# results = perform_inference("/path/to/images", model, WEIGHTS_PATH)
# for filename, pred_class in results:
#     print(f"Image: {filename}, Predicted Class: {pred_class}")
