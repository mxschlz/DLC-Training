import os
os.environ["QT_API"] = "pyqt5"
import shutil
import subprocess
import tarfile
import deeplabcut
import matplotlib
from pathlib import Path
from deeplabcut.utils import auxiliaryfunctions
matplotlib.use("TkAgg")

# On Windows, this might look like: r"C:\Users\Max\Desktop\ocapi-Max-...\config.yaml"
config_path = "/home/maxschulz/IPSY1-Storage/Projects/ac/Transfer/Max2Max/ocapi-Max-2026-02-13/config.yaml"

# Define which shuffle you want to work with (usually 1)
SHUFFLE_IDX = 1

# AUTOMATIC PATH FIX: Updates project_path in config.yaml to match current location
# This allows moving the project between Server (Linux) and Laptop (Windows) seamlessly.
cfg = auxiliaryfunctions.read_config(config_path)
current_project_path = str(Path(config_path).parents[0])
if cfg['project_path'] != current_project_path:
    cfg['project_path'] = current_project_path
    auxiliaryfunctions.write_config(config_path, cfg)

# Switch to TensorFlow engine
if cfg.get('engine') != 'tensorflow':
    cfg['engine'] = 'tensorflow'
    auxiliaryfunctions.write_config(config_path, cfg)

# CLEANUP: Remove old training data to force a fresh start
# This is necessary because create_training_dataset fails if folders exist,
# and train_network might try to resume from incompatible PyTorch checkpoints.
iteration = cfg['iteration']
train_ds_path = os.path.join(cfg['project_path'], 'training-datasets', f'iteration-{iteration}')
dlc_models_path = os.path.join(cfg['project_path'], 'dlc-models', f'iteration-{iteration}')

if os.path.exists(train_ds_path):
    print(f"Removing old training dataset: {train_ds_path}")
    shutil.rmtree(train_ds_path)
if os.path.exists(dlc_models_path):
    print(f"Removing old model directory: {dlc_models_path}")
    shutil.rmtree(dlc_models_path)

# PRE-CHECK: Manually download MobileNetV2 weights if missing (bypasses HTTP 403)
dlc_path = deeplabcut.__path__[0]
pretrained_path = os.path.join(dlc_path, 'pose_estimation_tensorflow', 'models', 'pretrained')
mobilenet_check_file = os.path.join(pretrained_path, 'mobilenet_v2_1.0_224.ckpt.index')

if not os.path.exists(mobilenet_check_file) or os.path.getsize(mobilenet_check_file) == 0:
    print(f"Downloading MobileNetV2 weights to {pretrained_path}...")
    os.makedirs(pretrained_path, exist_ok=True)
    # Try the official TensorFlow download URL which is often more accessible
    url = "http://download.tensorflow.org/models/mobilenet_v2_1.0_224.tgz"
    tgz_path = os.path.join(pretrained_path, 'mobilenet_v2_1.0_224.tgz')

    # Use curl with -f (fail) to prevent saving HTML error pages as files
    try:
        subprocess.run(["curl", "-f", "-L", "-A", "Mozilla/5.0", "-o", tgz_path, url], check=True)
    except subprocess.CalledProcessError:
        print("Curl failed. Attempting fallback to wget...")
        try:
            subprocess.run(["wget", "-U", "Mozilla/5.0", "-O", tgz_path, url], check=True)
        except subprocess.CalledProcessError:
            print(f"\nCRITICAL ERROR: Automatic download failed.\nPlease manually download: {url}\nAnd upload it to: {tgz_path}\nThen run this script again.")
            exit(1)

    try:
        with tarfile.open(tgz_path, mode='r:gz') as tar:
            tar.extractall(path=pretrained_path)
    except tarfile.ReadError:
        print("ERROR: Downloaded file is not a valid gzip. It might be an HTML error page.")
        with open(tgz_path, 'r', errors='ignore') as f:
            print(f"File content preview: {f.read(300)}")
        os.remove(tgz_path)
        raise RuntimeError("Failed to download MobileNet weights. Please check your internet connection or download manually.")
    
    if os.path.exists(tgz_path):
        os.remove(tgz_path)
    print("Download complete.")

# Uncomment the lines below only when you are completely done labeling!
deeplabcut.create_training_dataset(config_path, num_shuffles=1, net_type='resnet_50', augmenter_type='imgaug')

# Training is finished! Commenting out to prevent accidental re-runs.
deeplabcut.train_network(config_path, shuffle=SHUFFLE_IDX, displayiters=100, saveiters=1000, maxiters=50000)
deeplabcut.evaluate_network(config_path, Shuffles=[SHUFFLE_IDX], plotting=True)

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
    deeplabcut.filterpredictions(config_path, test_video)
    
    # This creates a video with the dots drawn on it
    # We use filtered=True to use the smoothed data
    deeplabcut.create_labeled_video(config_path, [test_video], draw_skeleton=True, filtered=True)

    # Plot trajectories to see if the filter fixed the jumps
    deeplabcut.plot_trajectories(config_path, [test_video], filtered=True)

    # OPTIONAL: Plot the likelihood (confidence) over time to spot "trashy" frames
    # This saves a plot named '..._likelihood.png' in the video folder
    deeplabcut.plot_trajectories(config_path, [test_video])

# --- EXPORT: Generate the model package for DeepLabCut-Live ---
deeplabcut.export_model(config_path, shuffle=SHUFFLE_IDX, make_tar=True)
