#!/usr/bin/env python3
"""Count the total number of trajectories in the processed dataset."""

import os
from tensorflow.python.lib.io import file_io

tfrecord_dataset_path = "gs://max-us-central2/susieprocessed_tfrecords_4step"

# Count trajectories in each split and scene
splits = ["training", "validation"]
scenes = {
    "training": ["A", "B", "C", "D"],
    "validation": ["D"]
}

total_count = 0
counts_by_split = {}

for split in splits:
    split_count = 0
    for scene in scenes[split]:
        scene_path = os.path.join(tfrecord_dataset_path, split, scene)
        try:
            # List all files in the scene directory
            files = file_io.list_directory(scene_path)
            # Count .tfrecord files
            tfrecord_files = [f for f in files if f.endswith('.tfrecord')]
            scene_count = len(tfrecord_files)
            split_count += scene_count
            print(f"{split}/{scene}: {scene_count} trajectories")
        except Exception as e:
            print(f"Error reading {scene_path}: {e}")
    
    counts_by_split[split] = split_count
    total_count += split_count
    print(f"\n{split.upper()} TOTAL: {split_count} trajectories")
    print("-" * 50)

print(f"\n{'='*50}")
print(f"GRAND TOTAL: {total_count} trajectories")
print(f"{'='*50}")
