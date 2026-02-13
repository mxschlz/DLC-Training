import os
os.environ["QT_API"] = "pyqt5"
import deeplabcut
import matplotlib
from pathlib import Path
from deeplabcut.utils import auxiliaryfunctions
matplotlib.use("TkAgg")

# On Windows, this might look like: r"C:\Users\Max\Desktop\ocapi-Max-...\config.yaml"
config_path = "/home/maxschulz/PycharmProjects/DLC-Training/ocapi-Max-2026-02-13/config.yaml"

# AUTOMATIC PATH FIX: Updates project_path in config.yaml to match current location
# This allows moving the project between Server (Linux) and Laptop (Windows) seamlessly.
cfg = auxiliaryfunctions.read_config(config_path)
current_project_path = str(Path(config_path).parents[0])
if cfg['project_path'] != current_project_path:
    cfg['project_path'] = current_project_path
    auxiliaryfunctions.write_config(config_path, cfg)

deeplabcut.label_frames(config_path)

# Uncomment the lines below only when you are completely done labeling!
# deeplabcut.create_training_dataset(config_path, num_shuffles=1, net_type='resnet_50', augmenter_type='imgaug')
#
# deeplabcut.train_network(config_path, shuffle=1, displayiters=100, saveiters=1000, maxiters=10000)
#
# deeplabcut.evaluate_network(config_path, Shuffles=[1], plotting=True)
