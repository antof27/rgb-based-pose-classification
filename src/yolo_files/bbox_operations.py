import os
from PIL import Image

WIDTH = 960
HEIGHT = 540

def calculate_weighted_score(box_info, score_weight=0.6, area_weight=0.4):
    box = box_info["box"]
    score = box_info["score"]
    x_min, y_min, x_max, y_max = box
    area = (x_max - x_min) * (y_max - y_min)
    normalized_area = area / (WIDTH * HEIGHT)
    return score_weight * score + area_weight * normalized_area

def enlarge_box(box):
    x_min, y_min, x_max, y_max = box
    width = x_max - x_min
    height = y_max - y_min
    enlargement = 0.1 * (width * height) ** 0.5  # 10% of the square root of the area
    #Clipped between 0 and 960 for x and 0 and 540 for y
    return [
        max(0, x_min - enlargement),
        max(0, y_min - enlargement),
        min(x_max + enlargement, WIDTH),
        min(y_max + enlargement, HEIGHT),
    ]

def interpolate_boxes(box1, box2, alpha):
    return [
        box1[i] + alpha * (box2[i] - box1[i])
        for i in range(len(box1))
    ]

def process_and_crop_images(all_predictions, images_folder, output_folder):
    """
    Processes the predictions from a dictionary, enlarges bounding boxes,
    and crops the images based on the bounding boxes. Cropped images are
    saved to the specified output folder.

    Args:
        all_predictions (dict): Dictionary containing predictions, where keys are image names
                                 and values are lists of bounding box information.
        images_folder (str): Folder containing the original images.
        output_folder (str): Folder where cropped images will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Keep only the best detection for each frame
    filtered_predictions = {}
    for key, boxes in all_predictions.items():
        if boxes:  # If there are detections
            best_box_info = max(boxes, key=lambda b: calculate_weighted_score(b))
            filtered_predictions[key] = [best_box_info]
        else:  # If no detections, set the full image box
            filtered_predictions[key] = [{"box": [0, 0, WIDTH, HEIGHT], "score": 1.00}]

    # Enlarge bounding boxes
    for key in filtered_predictions:
        for detection in filtered_predictions[key]:
            detection["box"] = enlarge_box(detection["box"])

    # Process each image and crop based on bounding boxes
    for image_name, boxes in filtered_predictions.items():
        image_path = os.path.join(images_folder, image_name)
        if MODE == "depth":
            image_path = image_path.replace(".jpg", ".png")
        
        if not os.path.exists(image_path):
            print(f"Image {image_name} not found in {images_folder}. Skipping.")
            continue

        try:
            with Image.open(image_path) as img:
                for i, box_info in enumerate(boxes):
                    box = box_info["box"]
                    # Extract bounding box coordinates
                    x_min, y_min, x_max, y_max = map(int, box)
                    cropped_img = img.crop((x_min, y_min, x_max, y_max))

                    output_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_{i}.jpg")
                    cropped_img.save(output_path)
                    print(f"Saved cropped image to {output_path}")
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    print(f"Finished processing and saving cropped images to {output_folder}")
