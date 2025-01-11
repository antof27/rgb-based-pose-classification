import os
import cv2
import json
from ultralytics import YOLOv10
import sys

def run_yolo_inference(weights_path, inference_images_path, images_output_path, conf_threshold=0.2):
    """
    Run YOLOv10 inference on the images in the specified directory and save the predictions to a dictionary.
    
    Args:
        trained_weights_path (str): Path to the trained YOLOv10 weights.
        validation_images_path (str): Directory containing the images to process.
        json_output_path (str): Path to the output JSON file for storing predictions.
        conf_threshold (float): Confidence threshold for YOLO predictions. Default is 0.2.
    """
    
    model = YOLOv10(weights_path)
    all_predictions = {}


    for image_file in os.listdir(inference_images_path):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(inference_images_path, image_file)
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read image {image_path}")
                continue

            
            inference_results = model.predict(image, conf=conf_threshold)

            boxes = inference_results[0].boxes.xyxy.cpu().tolist()  # Bounding box coordinates
            scores = inference_results[0].boxes.conf.cpu().tolist()  # Confidence scores

            all_predictions[image_file] = [{"box": box, "score": score} for box, score in zip(boxes, scores)]

    return all_predictions