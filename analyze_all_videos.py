import os
import multiprocessing
from pathlib import Path

# Force CPU inference - the GT 710 GPU is extremely slow and will bottleneck or OOM.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["QT_API"] = "pyqt5"

# --- CONFIGURATION ---
config_path = "/home/maxschulz/IPSY1-Storage/Projects/ac/Transfer/Max2Max/ocapi-Max-2026-02-13/config.yaml"
video_source = r"/home/maxschulz/IPSY1-Storage/Projects/ac/Experiments/running_studies/OCAPI/SMILE/Videos"

TARGET_ITERATION = 0 

def setup_config():
    import deeplabcut
    from deeplabcut.utils import auxiliaryfunctions
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

def process_video(video_path):
    # Import inside the worker process to avoid TF fork deadlocks
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["QT_API"] = "pyqt5"
    # Limit OpenMP threads to avoid oversubscription
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    
    import deeplabcut
    import matplotlib
    matplotlib.use("Agg")
    
    # Configure TensorFlow threads explicitly
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    print(f"--- Starting analysis for: {video_path} ---")
    try:
        deeplabcut.analyze_videos(config_path, [video_path], save_as_csv=True, TFGPUinference=False)
        deeplabcut.filterpredictions(config_path, [video_path])
        deeplabcut.create_labeled_video(config_path, [video_path], draw_skeleton=True, filtered=True)
        print(f"--- Completed: {video_path} ---")
    except Exception as e:
        print(f"--- Error processing {video_path}: {e} ---")

if __name__ == '__main__':
    # Use spawn to prevent TF/CUDA fork deadlocks
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
        
    setup_config()
    
    video_paths = [str(p) for p in Path(video_source).rglob('*') if p.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']]
    print(f"Found {len(video_paths)} videos to analyze.")
    
    # We have 48 CPU cores and ~188GB RAM. 
    # 12 workers will process 12 videos concurrently, maxing out CPU without overloading RAM.
    NUM_WORKERS = 12 
    print(f"Starting parallel processing with {NUM_WORKERS} workers...")
    
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        pool.map(process_video, video_paths)
        
    print("All videos processed!")