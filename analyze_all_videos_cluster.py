import os
from pathlib import Path

# IMPORTANT: Do NOT disable GPUs here since we want the cluster to use them!
os.environ["QT_API"] = "pyqt5"

import deeplabcut
from deeplabcut.utils import auxiliaryfunctions
import matplotlib
matplotlib.use("Agg")

# --- CONFIGURATION ---
# Note: Ensure these paths exist on the cluster! 
# If the cluster has a different file system, you will need to update these paths.
config_path = "/storage/personal/ocapi-Max-2026-02-13/config.yaml"
video_source = r"/storage/personal/Videos"

TARGET_ITERATION = 0 

def setup_config():
    cfg = auxiliaryfunctions.read_config(config_path)
    current_project_path = str(Path(config_path).parents[0])
    if cfg['project_path'] != current_project_path:
        print(f"Updating project_path in config to: {current_project_path}")
        cfg['project_path'] = current_project_path
        auxiliaryfunctions.write_config(config_path, cfg)

    print(f"Current config iteration: {cfg['iteration']}")

    if cfg['iteration'] != TARGET_ITERATION:
        print(f"Switching configuration to use Iteration {TARGET_ITERATION}...")
        cfg['iteration'] = TARGET_ITERATION
        auxiliaryfunctions.write_config(config_path, cfg)
    else:
        print(f"Configuration is already set to Iteration {TARGET_ITERATION}.")

if __name__ == '__main__':
    setup_config()
    
    # 1. Get All Videos
    video_paths = [str(p) for p in Path(video_source).rglob('*') if p.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']]
    print(f"Found {len(video_paths)} videos to analyze.")
    
    # 2. Run Analysis Pipeline (GPU Accelerated)
    # We pass the entire list to analyze_videos. DeepLabCut will use the powerful 
    # cluster GPU to crunch through them sequentially at high speed.
    print("Starting GPU analysis...")
    
    deeplabcut.analyze_videos(
        config_path, 
        video_paths, 
        save_as_csv=True, 
        batchsize=32,          # High batch size for GPU efficiency
        TFGPUinference=True,   # Ensure GPU is used
        allow_growth=True
    )
    
    print("Filtering predictions...")
    deeplabcut.filterpredictions(config_path, video_paths)
    
    print("Creating labeled videos...")
    deeplabcut.create_labeled_video(config_path, video_paths, draw_skeleton=True, filtered=True)
    
    print("All videos processed on cluster!")
