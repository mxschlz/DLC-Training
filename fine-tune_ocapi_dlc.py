import os
os.environ["QT_API"] = "pyqt5"
import deeplabcut
import matplotlib
from pathlib import Path
from deeplabcut.utils import auxiliaryfunctions
matplotlib.use("TkAgg")

# On Windows, this might look like: r"C:\Users\Max\Desktop\ocapi-Max-...\config.yaml"
config_path = "/home/maxschulz/IPSY1-Storage/Projects/ac/Transfer/Max2Max/ocapi-Max-2026-02-13/config.yaml"

# AUTOMATIC PATH FIX: Updates project_path in config.yaml to match current location
# This allows moving the project between Server (Linux) and Laptop (Windows) seamlessly.
cfg = auxiliaryfunctions.read_config(config_path)
current_project_path = str(Path(config_path).parents[0])
if cfg['project_path'] != current_project_path:
    cfg['project_path'] = current_project_path
    auxiliaryfunctions.write_config(config_path, cfg)

#deeplabcut.label_frames(config_path)

# Uncomment the lines below only when you are completely done labeling!
#deeplabcut.create_training_dataset(config_path, num_shuffles=1, net_type='resnet_50', augmenter_type='imgaug')

# Training is finished! Commenting out to prevent accidental re-runs.
# deeplabcut.train_network(config_path, shuffle=1, displayiters=100, saveiters=1000, maxiters=50000)
# deeplabcut.evaluate_network(config_path, Shuffles=[1], plotting=True)

# --- QUALITY CONTROL: Analyze one video to check for jitter/swaps ---
# Get the list of videos from the config file
cfg = auxiliaryfunctions.read_config(config_path)
video_list = list(cfg['video_sets'].keys())

if video_list:
    test_video = video_list[0] # Pick the first video to test
    print(f"Analyzing test video: {test_video}")
    
    # save_as_csv=True is useful if you want to inspect confidence scores later
    deeplabcut.analyze_videos(config_path, [test_video], save_as_csv=True)
    
    # Filter predictions to remove large jumps (fixes the high RMSE issue)
    deeplabcut.filterpredictions(config_path, [test_video])
    
    # This creates a video with the dots drawn on it
    # We use filtered=True to use the smoothed data
    deeplabcut.create_labeled_video(config_path, [test_video], draw_skeleton=True, filtered=True)

    # Plot trajectories to see if the filter fixed the jumps
    deeplabcut.plot_trajectories(config_path, [test_video], filtered=True)

    # OPTIONAL: Plot the likelihood (confidence) over time to spot "trashy" frames
    # This saves a plot named '..._likelihood.png' in the video folder
    deeplabcut.plot_trajectories(config_path, [test_video], show_figures=False, plot_likelihood=True)

# --- EXPORT: Generate the model package for DeepLabCut-Live ---
# This creates a folder (and .tar.gz) in 'exported-models' inside your project directory
deeplabcut.export_model(config_path, shuffle=1, make_tar=True)
