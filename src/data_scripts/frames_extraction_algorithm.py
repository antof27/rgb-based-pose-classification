import cv2
import os

# Set the directory paths
video_folder = '/media/coloranto/Volume/isDepthMap/data/dataset_video'
output_folder = '/media/coloranto/Volume/isDepthMap/data/dataset_frames'

# Set frames per second (fps) for extraction
fps = 24

# Make sure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through each video in the folder
for video_file in os.listdir(video_folder):
    if video_file.endswith('.mp4') or video_file.endswith('.avi'):  # Add other video extensions if needed
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Get the original fps of the video
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = int(original_fps / fps)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save the frame at the specified fps
            if frame_count % frame_interval == 0:
                frame_filename = f"{video_name}_{frame_count:05d}.jpg"
                frame_path = os.path.join(output_folder, frame_filename)
                cv2.imwrite(frame_path, frame)
            
            frame_count += 1
        
        # Release the video capture
        cap.release()

print("Frames extracted and saved successfully!")
