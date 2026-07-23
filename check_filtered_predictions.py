import os
from pathlib import Path

video_source = r"/home/maxschulz/IPSY1-Storage/Projects/ac/Experiments/running_studies/OCAPI/all_videos_combined"

def check_filtered_predictions():
    # 1. Get all video paths
    video_paths = [p for p in Path(video_source).glob('*') if p.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']]
    
    missing_filtered = []
    found_filtered = []

    # 2. Check for filtered files
    for video_path in video_paths:
        # DLC appends the scorer and 'filtered.h5' to the original video's base name
        filtered_files = list(video_path.parent.glob(f"{video_path.stem}*filtered.h5"))
        
        if filtered_files:
            found_filtered.append(video_path.name)
        else:
            missing_filtered.append(video_path.name)
            
    # 3. Print the summary report
    print(f"Total videos found: {len(video_paths)}")
    print(f"Videos WITH filtered predictions: {len(found_filtered)}")
    print(f"Videos MISSING filtered predictions: {len(missing_filtered)}")
    
    if missing_filtered:
        print("\n--- Missing Filtered Predictions For ---")
        for name in missing_filtered:
            print(f" - {name}")

if __name__ == "__main__":
    check_filtered_predictions()