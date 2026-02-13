import deeplabcut
from pathlib import Path

# Path to the folder containing the collected videos
video_source = r"G:\Meine Ablage\PhD\data\OCAPI\all_videos_combined"

# Generate a list of all video file paths in that directory
video_paths = [str(p) for p in Path(video_source).glob('*') if p.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']]

config_path = deeplabcut.create_new_project('ocapi', 'Max',
             video_paths,
             copy_videos=True)
