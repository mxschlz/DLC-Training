import os
import shutil
import subprocess
from pathlib import Path

def collect_videos(source_root, dest_folder, extensions=('.mp4', '.avi', '.mov', '.mkv')):
    """
    Walks through source_root, finds all video files, and copies them to dest_folder.
    """
    source_path = Path(source_root)
    dest_path = Path(dest_folder)
    
    # Create destination if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Searching for videos in: {source_path}")
    copied_count = 0
    
    for root, dirs, files in os.walk(source_path):
        # Skip the 'original' directory if it exists at this level
        if 'original' in dirs:
            dirs.remove('original')

        for file in files:
            if file.lower().endswith(extensions):
                src_file = os.path.join(root, file)
                # Force .mp4 extension for the destination
                dst_filename = Path(file).stem + ".mp4"
                dst_file = dest_path / dst_filename
                
                # Simple collision handling: prepend parent folder name if file exists
                if dst_file.exists():
                    print(f"File {dst_filename} exists in destination. Renaming to avoid overwrite.")
                    parent_name = Path(root).name
                    dst_file = dest_path / f"{parent_name}_{dst_filename}"
                
                print(f"Converting {file} to {dst_file.name} with rotation metadata...")
                # Remux to mp4 without re-encoding (very fast) and add rotation flag
                cmd = [
                    'ffmpeg', '-y', '-v', 'error',
                    '-i', src_file,
                    '-c', 'copy',
                    '-metadata:s:v:0', 'rotate=180',
                    str(dst_file)
                ]
                subprocess.run(cmd, check=True)
                copied_count += 1
                
    print(f"Done! Copied {copied_count} videos to {dest_path}")

if __name__ == "__main__":
    # Paths based on your request
    SOURCE_DIR = r"G:\Meine Ablage\PhD\data\OCAPI\input"
    DEST_DIR = r"G:\Meine Ablage\PhD\data\OCAPI\all_videos_combined"
    
    collect_videos(SOURCE_DIR, DEST_DIR)