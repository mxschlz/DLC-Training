import os
os.environ["QT_API"] = "pyqt5"
import deeplabcut
from pathlib import Path
from deeplabcut.utils import auxiliaryfunctions
import matplotlib
matplotlib.use("Agg") # Agg is safer for headless frame extraction

# Path to the folder containing the collected videos
video_source = r"/home/maxschulz/IPSY1-Storage/Projects/ac/Experiments/running_studies/OCAPI/all_videos_combined"

# Generate a list of all video file paths in that directory
video_paths = [str(p) for p in Path(video_source).glob('*') if p.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']]

config_path = deeplabcut.create_new_project('ocapi', 'Max',
             video_paths,
             copy_videos=False)

# Define the body parts you want to label
cfg = auxiliaryfunctions.read_config(config_path)
cfg['bodyparts'] = [
            'left_iris_center', 'right_iris_center',
            'left_eye_outer_corner', 'right_eye_outer_corner',
            'nose_tip', 'chin',
            'left_mouth_corner', 'right_mouth_corner'
        ]  # Add your specific labels here
auxiliaryfunctions.write_config(config_path, cfg)

# 'uniform' is fastest; 'kmeans' is smartest.
deeplabcut.extract_frames(config_path, mode='automatic', algo='kmeans', userfeedback=False, crop=False, cluster_step=100)

print("-" * 50)
print(f"Project created successfully!\nConfig path: {config_path}")
print("Please copy this path into fine-tune_ocapi_dlc.py")
print("-" * 50)
