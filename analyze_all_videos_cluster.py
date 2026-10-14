import os
import math
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
config_path = "/home/maxschulz/ocapi-Max-2026-02-13/config.yaml"
video_source = "/home/maxschulz/Videos"

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
    
    video_paths = [str(p) for p in Path(video_source).rglob('*') if p.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv'] and '@eaDir' not in p.parts]
    video_paths.sort() # Ensure consistent ordering across all cluster nodes
    
    # Split videos across GPUs if running as a Slurm Array Job
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    task_count = os.environ.get('SLURM_ARRAY_TASK_COUNT')
    
    if task_id is not None and task_count is not None:
        task_id = int(task_id)
        task_count = int(task_count)
        chunk_size = math.ceil(len(video_paths) / task_count)
        start_idx = task_id * chunk_size
        end_idx = start_idx + chunk_size
        video_paths = video_paths[start_idx:end_idx]
        print(f"SLURM Array Task {task_id}: Assigned {len(video_paths)} videos (indices {start_idx} to {end_idx}).")
    else:
        print(f"Found {len(video_paths)} videos to analyze (running single node).")
    
    # Filter out already analyzed videos
    unprocessed_videos = []
    for vid_path in video_paths:
        vid_dir = Path(vid_path).parent
        vid_name = Path(vid_path).stem
        # Check if an .h5 file containing the video name exists in the same directory
        h5_files = list(vid_dir.glob(f"{vid_name}*.h5"))
        if not h5_files:
            unprocessed_videos.append(vid_path)
            
    print(f"Skipping {len(video_paths) - len(unprocessed_videos)} already analyzed videos.")
    print(f"Remaining videos to analyze: {len(unprocessed_videos)}")
    
    if len(unprocessed_videos) == 0:
        print("All videos are already analyzed! Skipping analysis step.")
    else:
        print("Starting GPU analysis...")
        deeplabcut.analyze_videos(
            config_path, 
            unprocessed_videos, 
            save_as_csv=True, 
            batchsize=32,          # High batch size for GPU efficiency
            TFGPUinference=True,   # Ensure GPU is used
            allow_growth=True
        )

    
    # 2. Run Analysis Pipeline (GPU Accelerated)
    # We pass the entire list to analyze_videos. DeepLabCut will use the powerful 
    # cluster GPU to crunch through them sequentially at high speed.
    print("Filtering predictions...")
    for vid in video_paths:
        try:
            deeplabcut.filterpredictions(config_path, [vid], filtertype='median')
        except Exception as e:
            print(f"Error filtering {vid}: {e}")
            
    print("Creating labeled videos...")
    for vid in video_paths:
        try:
            deeplabcut.create_labeled_video(config_path, [vid], draw_skeleton=True, filtered=True)
        except Exception as e:
            print(f"Error labeling {vid}: {e}")
    
    print("All videos processed on cluster!")
