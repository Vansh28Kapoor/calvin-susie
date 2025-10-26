import numpy as np
import tensorflow as tf 
from tqdm import tqdm 
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tensorflow.python.lib.io import file_io

# Add the project directory to the path
sys.path.insert(0, "/nfs/aidm_nfs/vansh/code/calvin-susie/calvin_models")

# Import the SUSIE model
from calvin_agent.evaluation.jax_diffusion_model import DiffusionModel

########## Dataset paths ###########
raw_dataset_path = "gs://max-us-central2/unzipped_files/task_ABC_D"
tfrecord_dataset_path = "gs://max-us-central2/susieprocessed_tfrecords"

"""Timer utility."""

import time
from collections import defaultdict


class _TimerContextManager:
    def __init__(self, timer: "Timer", key: str):
        self.timer = timer
        self.key = key

    def __enter__(self):
        self.timer.tick(self.key)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.timer.tock(self.key)


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.counts = defaultdict(int)
        self.times = defaultdict(float)
        self.start_times = {}

    def tick(self, key):
        if key in self.start_times:
            raise ValueError(f"Timer is already ticking for key: {key}")
        self.start_times[key] = time.time()

    def tock(self, key):
        if key not in self.start_times:
            raise ValueError(f"Timer is not ticking for key: {key}")
        self.counts[key] += 1
        self.times[key] += time.time() - self.start_times[key]
        del self.start_times[key]

    def context(self, key):
        """
        Use this like:

        with timer.context("key"):
            # do stuff

        Then timer.tock("key") will be called automatically.
        """
        return _TimerContextManager(self, key)

    def get_total_times(self, reset=True):
        ret = {key: self.times[key] for key in self.times}
        if reset:
            self.reset()
        return ret

    def get_average_times(self, reset=True):
        ret = {key: self.times[key] / self.counts[key] for key in self.counts}
        if reset:
            self.reset()
        return ret


timer = Timer()

########## SUSIE Model Initialization ###########
print("Initializing SUSIE model for goal image generation...")
checkpoint_path = "/nfs/aidm_nfs/vansh/code/calvin-susie/checkpoints/diffusion_model"
os.environ["DIFFUSION_MODEL_CHECKPOINT"] = checkpoint_path
susie_model = DiffusionModel()

########## Helper functions for GCS operations ###########
def gcs_exists(path):
    """Check if a GCS path exists"""
    try:
        file_io.stat(path)
        return True
    except:
        return False

def gcs_makedirs(path):
    """Create directory structure in GCS by creating a placeholder file"""
    if not gcs_exists(path):
        placeholder_path = os.path.join(path, ".placeholder")
        with file_io.FileIO(placeholder_path, 'w') as f:
            f.write("")

def gcs_load_npy(gcs_path, allow_pickle=False):
    """Load numpy file from GCS"""
    with file_io.FileIO(gcs_path, 'rb') as f:
        return np.load(f, allow_pickle=allow_pickle)

def gcs_load_npz(gcs_path):
    """Load npz file from GCS"""
    with file_io.FileIO(gcs_path, 'rb') as f:
        return np.load(f)

def load_single_episode(ep_id_str, split):
    """Load a single episode file from GCS"""
    episode_path = os.path.join(raw_dataset_path, split, "episode_" + ep_id_str + ".npz")
    try:
        timestep_data = gcs_load_npz(episode_path)
        return {
            'rel_actions': timestep_data["rel_actions"],
            'robot_obs': timestep_data["robot_obs"],
            'rgb_static': timestep_data["rgb_static"]
        }
    except Exception as e:
        print(f"Error loading episode {ep_id_str}: {e}")
        return None

########## Main logic ###########
# Create directory structure in GCS
directories_to_create = [
    tfrecord_dataset_path,
    os.path.join(tfrecord_dataset_path, "training"),
    os.path.join(tfrecord_dataset_path, "validation"),
    os.path.join(tfrecord_dataset_path, "training/A"),
    os.path.join(tfrecord_dataset_path, "training/B"),
    os.path.join(tfrecord_dataset_path, "training/C"),
    os.path.join(tfrecord_dataset_path, "training/D"),
    os.path.join(tfrecord_dataset_path, "validation/D")
]

print("Creating directory structure in GCS...")
for directory in directories_to_create:
    gcs_makedirs(directory)

def make_seven_characters(id):
    id = str(id)
    while len(id) < 7:
        id = "0" + id
    return id

def tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )

def string_to_feature(str_value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[str_value.encode("UTF-8")])
    )

# Fixed batch size for SUSIE to prevent JIT recompilation
SUSIE_BATCH_SIZE = 130

class SusieGoalBatcher:
    """
    Manages batching of image sequences from multiple trajectories to efficiently
    generate SUSIE goals with minimal padding. Supports sliding window processing.
    """
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size
        self.image_buffer = []
        self.lang_buffer = []
        self.source_buffer = []  # Stores (traj_idx, img_idx_in_traj)
        self.all_generated_goals = {}
        
    def clear_buffers(self):
        """Clear all buffers - useful for sliding window processing"""
        self.image_buffer = []
        self.lang_buffer = []
        self.source_buffer = []

    def add_trajectory(self, image_sequence, language_instruction, trajectory_index):
        """Adds a trajectory's data to the buffers and processes full batches."""
        seq_len = len(image_sequence)
        
        # More efficient: avoid list comprehension for source_info
        self.image_buffer.extend(image_sequence)
        self.lang_buffer.extend([language_instruction] * seq_len)
        self.source_buffer.extend((trajectory_index, i) for i in range(seq_len))
        
        # Process as many full batches as possible
        while len(self.image_buffer) >= self.batch_size:
            self._process_one_batch()

    def _process_one_batch(self, is_final_batch=False):
        """Processes one batch of images."""
        if not self.image_buffer:
            return

        if is_final_batch:
            actual_batch_size = len(self.image_buffer)
            image_batch = np.array(self.image_buffer)
            lang_batch = self.lang_buffer
            source_batch = self.source_buffer
            
            # Clear buffers
            self.image_buffer, self.lang_buffer, self.source_buffer = [], [], []
            
            # Pad for the final batch
            padded_image_batch = np.zeros((self.batch_size, *image_batch.shape[1:]), dtype=image_batch.dtype)
            padded_image_batch[:actual_batch_size] = image_batch
            padded_lang_batch = lang_batch + [lang_batch[-1]] * (self.batch_size - actual_batch_size)
            
            generated_goals = self.model.generate_batch(padded_lang_batch, padded_image_batch)[:actual_batch_size]
        else:
            # Process a full batch - convert to numpy array once
            image_batch = np.array(self.image_buffer[:self.batch_size])
            lang_batch = self.lang_buffer[:self.batch_size]
            source_batch = self.source_buffer[:self.batch_size]
            
            # Trim buffers
            self.image_buffer = self.image_buffer[self.batch_size:]
            self.lang_buffer = self.lang_buffer[self.batch_size:]
            self.source_buffer = self.source_buffer[self.batch_size:]
            
            generated_goals = self.model.generate_batch(lang_batch, image_batch)

        # Store the generated goals efficiently using vectorized operations
        # Group by trajectory to minimize array assignments
        traj_groups = {}
        for i, (traj_idx, img_idx) in enumerate(source_batch):
            if traj_idx not in traj_groups:
                traj_groups[traj_idx] = ([], [])
            traj_groups[traj_idx][0].append(img_idx)
            traj_groups[traj_idx][1].append(i)
        
        # Batch assign goals per trajectory
        for traj_idx, (img_indices, goal_indices) in traj_groups.items():
            self.all_generated_goals[traj_idx][img_indices] = generated_goals[goal_indices]

    def flush(self):
        """Processes any remaining images in the buffer."""
        if self.image_buffer:
            print(f"Flushing remaining {len(self.image_buffer)} images...")
            self._process_one_batch(is_final_batch=True)

    def get_goals_for_trajectory(self, trajectory_index):
        return self.all_generated_goals.get(trajectory_index)

def load_trajectory_data(function_data, max_io_workers=65):
    """
    Loads all data for a single trajectory from GCS in parallel.
    Does not generate SUSIE goals.
    """
    global raw_dataset_path
    idx_range, letter, ctr, split, lang_ann = function_data
    unique_pid = f"{split}_{letter}_{ctr}"

    start_id, end_id = idx_range[0], idx_range[1]
    episode_ids = list(range(start_id, end_id + 1))
    
    ep_id_strs = [make_seven_characters(ep_id) for ep_id in episode_ids]
    
    episode_data = [None] * len(ep_id_strs)
    
    with ThreadPoolExecutor(max_workers=max_io_workers) as executor:
        future_to_idx = {executor.submit(load_single_episode, ep_id_str, split): i for i, ep_id_str in enumerate(ep_id_strs)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                if result is not None:
                    episode_data[idx] = result
                else:
                    print(f"Failed to load episode {ep_id_strs[idx]} in trajectory {unique_pid}")
                    return None
            except Exception as e:
                print(f"Exception loading episode {ep_id_strs[idx]} in trajectory {unique_pid}: {e}")
                return None

    if any(data is None for data in episode_data):
        return None

    traj_rel_actions = np.array([ep['rel_actions'] for ep in episode_data], dtype=np.float32)
    traj_robot_obs = np.array([ep['robot_obs'] for ep in episode_data], dtype=np.float32)
    traj_rgb_static = np.array([ep['rgb_static'] for ep in episode_data], dtype=np.uint8)
    
    return {
        "function_data": function_data,
        "actions": traj_rel_actions,
        "proprioceptive_states": traj_robot_obs,
        "image_states": traj_rgb_static,
    }

def write_tfrecord(trajectory_data, susie_goals):
    """Writes a single trajectory with its SUSIE goals to a TFRecord in GCS."""
    global tfrecord_dataset_path
    idx_range, letter, ctr, split, lang_ann = trajectory_data["function_data"]
    
    output_dir = os.path.join(tfrecord_dataset_path, split, letter)
    output_path = os.path.join(output_dir, f"traj{ctr}.tfrecord")

    with tf.io.TFRecordWriter(output_path) as writer:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "actions": tensor_feature(trajectory_data["actions"]),
                    "proprioceptive_states": tensor_feature(trajectory_data["proprioceptive_states"]),
                    "image_states": tensor_feature(trajectory_data["image_states"]),
                    "language_annotation": string_to_feature(lang_ann),
                    "susie_goal_images": tensor_feature(susie_goals)
                }
            )
        )
        writer.write(example.SerializeToString())
    return output_path

# Let's prepare the inputs
function_inputs = []

# First let's do the train data
auto_lang_ann_path = os.path.join(raw_dataset_path, "training", "lang_annotations", "auto_lang_ann.npy")
auto_lang_ann = gcs_load_npy(auto_lang_ann_path, allow_pickle=True)
auto_lang_ann = auto_lang_ann.item()
all_language_annotations = auto_lang_ann["language"]["ann"]
idx_ranges = auto_lang_ann["info"]["indx"]

scene_info_path = os.path.join(raw_dataset_path, "training", "scene_info.npy")
scene_info = gcs_load_npy(scene_info_path, allow_pickle=True)
scene_info = scene_info.item()

A_ctr, B_ctr, C_ctr, D_ctr = 0, 0, 0, 0
for i, idx_range in enumerate(idx_ranges):
    start_idx = idx_range[0]
    if "calvin_scene_D" in scene_info and start_idx <= scene_info["calvin_scene_D"][1]:
        ctr = D_ctr
        D_ctr += 1
        letter = "D"
    elif start_idx <= scene_info["calvin_scene_B"][1]:
        ctr = B_ctr
        B_ctr += 1
        letter = "B"
    elif start_idx <= scene_info["calvin_scene_C"][1]:
        ctr = C_ctr
        C_ctr += 1
        letter = "C"
    else:
        ctr = A_ctr
        A_ctr += 1
        letter = "A"

    function_inputs.append((idx_range, letter, ctr, "training", all_language_annotations[i]))

print("Loading validation data annotations...")
# Next let's do the validation data
auto_lang_ann_path = os.path.join(raw_dataset_path, "validation", "lang_annotations", "auto_lang_ann.npy")
auto_lang_ann = gcs_load_npy(auto_lang_ann_path, allow_pickle=True)
auto_lang_ann = auto_lang_ann.item()
all_language_annotations = auto_lang_ann["language"]["ann"]
idx_ranges = auto_lang_ann["info"]["indx"]

print("Processing validation trajectories...")
ctr = 0
for i, idx_range in enumerate(idx_ranges):
    function_inputs.append((idx_range, "D", ctr, "validation", all_language_annotations[i]))
    ctr += 1

# Main processing logic with sliding window approach
print("Starting memory-efficient sliding window processing...")

# Configuration for sliding window
WINDOW_SIZE = 50  # Number of trajectories to keep in memory at once
LOAD_WORKERS = 50  # Workers for loading trajectories
WRITE_WORKERS = 80  # Workers for writing TFRecords

def estimate_memory_usage(trajectories_data):
    """Estimate memory usage for a list of trajectory data"""
    total_bytes = 0
    for traj in trajectories_data:
        if traj:
            total_bytes += traj["actions"].nbytes
            total_bytes += traj["proprioceptive_states"].nbytes  
            total_bytes += traj["image_states"].nbytes
            # Double for SUSIE goals (same shape as image_states)
            total_bytes += traj["image_states"].nbytes
    return total_bytes / (1024 * 1024)  # Convert to MB

def process_sliding_window(trajectory_inputs, window_size=WINDOW_SIZE):
    """
    Process trajectories in sliding windows to manage memory usage while 
    maintaining efficient batch utilization without padding.
    """
    processed_files = []
    total_trajectories = len(trajectory_inputs)
    
    # Process trajectories in windows  
    for window_start in tqdm(range(0, total_trajectories, window_size), desc="Processing windows"):
        window_end = min(window_start + window_size, total_trajectories)
        window_inputs = trajectory_inputs[window_start:window_end]
        is_final_window = window_end == total_trajectories
        
        print(f"\nProcessing window {window_start//window_size + 1}: trajectories {window_start}-{window_end-1}")
        
        # Step 1: Load current window of trajectories
        print("Loading trajectories...")
        loaded_trajectories = []
        timer.tick("window_load_trajectory")
        with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as executor:
            future_to_input = {executor.submit(load_trajectory_data, fi): fi for fi in window_inputs}
            for future in as_completed(future_to_input):
                result = future.result()
                if result:
                    loaded_trajectories.append(result)
        
        if not loaded_trajectories:
            print("No trajectories loaded in this window, skipping...")
            continue
        timer.tock("window_load_trajectory")
        print(timer.get_total_times())
        # Step 2: Process this window with a fresh batcher
        window_batcher = SusieGoalBatcher(susie_model, SUSIE_BATCH_SIZE)
        
        # Pre-allocate goals for this window
        for i, traj_data in enumerate(loaded_trajectories):
            traj_len = len(traj_data["image_states"])
            goal_shape = traj_data["image_states"].shape
            window_batcher.all_generated_goals[i] = np.zeros(goal_shape, dtype=np.uint8)
        
        # Estimate memory usage for monitoring
        memory_mb = estimate_memory_usage(loaded_trajectories)
        print(f"Window memory usage: ~{memory_mb:.1f} MB")
        
        # Step 3: Add all trajectories in this window to the batcher
        print(f"Processing SUSIE goals for {len(loaded_trajectories)} trajectories...")
        for i, traj_data in enumerate(loaded_trajectories):
            lang_ann = traj_data["function_data"][-1]
            window_batcher.add_trajectory(traj_data["image_states"], lang_ann, i)
        
        # Step 4: Flush any remaining images in this window
        if is_final_window or len(window_batcher.image_buffer) > 0:
            print("Flushing remaining images in window...")
            window_batcher.flush()
        
        # Step 5: Write all TFRecords for this window
        print(f"Writing TFRecords for window...")
        window_processed = []
        with ThreadPoolExecutor(max_workers=WRITE_WORKERS) as executor:
            future_to_data = {}
            for i, traj_data in enumerate(loaded_trajectories):
                goals = window_batcher.get_goals_for_trajectory(i)
                if goals is not None:
                    future = executor.submit(write_tfrecord, traj_data, goals)
                    future_to_data[future] = traj_data["function_data"]
            
            for future in as_completed(future_to_data):
                try:
                    path = future.result()
                    window_processed.append(path)
                except Exception as e:
                    function_data = future_to_data[future]
                    print(f"Error writing TFRecord for trajectory {function_data}: {e}")
        
        processed_files.extend(window_processed)
        
        # Clear memory for next window
        del loaded_trajectories
        del window_batcher
        
        print(f"Window complete. Window processed: {len(window_processed)}, Total processed: {len(processed_files)}")
    
    return processed_files

# Execute the sliding window processing
processed_files = process_sliding_window(function_inputs)

print(f"\nProcessing complete! Successfully wrote {len(processed_files)} TFRecords.")
