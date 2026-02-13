import os
os.environ["QT_API"] = "pyqt5"
import deeplabcut
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")

# Path to the folder containing the collected videos
video_source = r"C:\Users\Max\Desktop\all_videos_combined\flipped"

# Generate a list of all video file paths in that directory
video_paths = [str(p) for p in Path(video_source).glob('*') if p.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']]

config_path = deeplabcut.create_new_project('ocapi', 'Max',
             video_paths,
             copy_videos=True)

# 'uniform' is fastest; 'kmeans' is smartest.
deeplabcut.extract_frames(config_path, mode='automatic', algo='uniform', userfeedback=False, crop="GUI", cluster_step=20)

deeplabcut.label_frames(config_path)