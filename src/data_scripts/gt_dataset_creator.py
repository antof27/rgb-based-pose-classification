import pandas as pd
import os
from glob import glob


annotations_csv = "/media/coloranto/Volume/isDepthMap/data/dataset_video.csv"
images_folder = "/media/coloranto/Volume/isDepthMap/data/dataset_frames"      
output_csv = "/media/coloranto/Volume/isDepthMap/data/frames_annotation.csv"  

def read_csv(csv_path):
    return pd.read_csv(csv_path)


annotations_df = read_csv(annotations_csv)

frames = glob(os.path.join(images_folder, "*.jpg"))  
frames.sort()  
frame_filenames = [os.path.splitext(os.path.basename(frame))[0] for frame in frames]

output_data = []

# One-hot encode skills: get unique skills and create a dictionary mapping skill IDs to one-hot encodings
unique_skills = annotations_df['id_skill'].unique()
skill_mapping = {skill: idx for idx, skill in enumerate(unique_skills)}
one_hot_labels = pd.get_dummies(annotations_df['id_skill']).columns.tolist()

# Process each frame and determine its skill ID
for filename in frame_filenames:
    video_id, frame_number = filename.rsplit("_", 1)
    frame_number = int(frame_number)
    
    # associate skill with frame
    skill_row = annotations_df[(annotations_df['id_video'] == video_id) &
                               (annotations_df['start_skill_frame'] <= frame_number) &
                               (annotations_df['end_skill_frame'] >= frame_number)]
    
    #check the range
    if not skill_row.empty:
        id_skill = skill_row.iloc[0]['id_skill']
    else:
        id_skill = "none"
    
    # Create one-hot encoding for the skill
    skill_vector = [0] * len(one_hot_labels)
    if id_skill != "none":
        skill_vector[skill_mapping[id_skill]] = 1
    
    # Append data to output list
    output_data.append([filename] + skill_vector)


columns = ["filename"] + one_hot_labels
output_df = pd.DataFrame(output_data, columns=columns)
output_df.to_csv(output_csv, index=False)
