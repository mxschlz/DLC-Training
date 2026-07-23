import os
os.environ["QT_API"] = "pyqt5"
import deeplabcut
from pathlib import Path
from deeplabcut.utils import auxiliaryfunctions
import matplotlib
matplotlib.use("Agg") # Safe for server (headless)

# --- CONFIGURATION ---
config_path = "/home/maxschulz/IPSY1-Storage/Projects/ac/Transfer/Max2Max/ocapi-Max-2026-02-13/config.yaml"
video_source = r"/home/maxschulz/IPSY1-Storage/Projects/ac/Experiments/running_studies/OCAPI/all_videos_combined"

# Set this to the iteration number of your ResNet50 model.
# Check your 'dlc-models' folder to be sure (e.g., iteration-0 is usually the first one).
TARGET_ITERATION = 0 

# --- 1. Setup & Portability Fixes ---
# Ensure the project path in config.yaml matches the current machine
cfg = auxiliaryfunctions.read_config(config_path)
current_project_path = str(Path(config_path).parents[0])
if cfg['project_path'] != current_project_path:
    print(f"Updating project_path in config to: {current_project_path}")
    cfg['project_path'] = current_project_path
    auxiliaryfunctions.write_config(config_path, cfg)

print(f"Current config iteration: {cfg['iteration']}")

if cfg['iteration'] != TARGET_ITERATION:
    print(f"Switching configuration to use Iteration {TARGET_ITERATION} (ResNet50)...")
    cfg['iteration'] = TARGET_ITERATION
    auxiliaryfunctions.write_config(config_path, cfg)
else:
    print(f"Configuration is already set to Iteration {TARGET_ITERATION}.")

# --- 2. Get All Videos ---
video_paths = [str(p) for p in Path(video_source).glob('*') if p.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']]
print(f"Found {len(video_paths)} videos to analyze.")

# --- 3. Run Analysis Pipeline ---
# Analyze (generates .h5 and .csv files)
# Optimization: Set batch_size=32 (or 16 if OOM errors occur) to maximize GPU utilization
deeplabcut.analyze_videos(config_path, video_paths, save_as_csv=True, batchsize=32, TFGPUinference=True, allow_growth=True)

# Filter (smooths out jitter)
deeplabcut.filterpredictions(config_path, video_paths)

# Create Video (overlays dots on the video)
# We use filtered=True to visualize the smoothed data
deeplabcut.create_labeled_video(config_path, video_paths, draw_skeleton=True, filtered=True)