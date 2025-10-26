import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
import sys
from multiprocessing import Pool
from tensorflow.python.lib.io import file_io
import tempfile
from PIL import Image
import jax
import jax.numpy as jnp

# Add the project directory to the path
sys.path.insert(0, "/nfs/aidm_nfs/vansh/code/calvin-susie/calvin_models")

# Import the SUSIE model
from calvin_agent.evaluation.jax_diffusion_model import DiffusionModel

########## Dataset paths ###########
raw_dataset_path = "gs://max-us-central2/unzipped_files/task_ABC_D"
tfrecord_dataset_path = "gs://max-us-central2/susieprocessed_tfrecords"

########## SUSIE Model Initialization ###########
print("Initializing SUSIE model for goal image generation...")
checkpoint_path = "/nfs/aidm_nfs/vansh/code/calvin-susie/checkpoints/diffusion_model"
os.environ["DIFFUSION_MODEL_CHECKPOINT"] = checkpoint_path
susie_model = DiffusionModel()

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
        with file_io.FileIO(placeholder_path, "w") as f:
            f.write("")


def gcs_load_npy(gcs_path, allow_pickle=False):
    """Load numpy file from GCS"""
    with file_io.FileIO(gcs_path, "rb") as f:
        return np.load(f, allow_pickle=allow_pickle)


def gcs_load_npz(gcs_path):
    """Load npz file from GCS"""
    with file_io.FileIO(gcs_path, "rb") as f:
        return np.load(f)


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
    os.path.join(tfrecord_dataset_path, "validation/D"),
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


# def generate_susie_goals(image_sequence, language_instruction):
#     """Generate SUSIE goal images for each observation in the trajectory."""

#     goals = []
#     seq_len = len(image_sequence)

#     for i, obs_image in enumerate(image_sequence):
#         try:
#             # Generate goal image using SUSIE
#             goal_image = susie_model.generate(language_instruction, obs_image)

#             # Ensure the goal image is in the correct format (200x200x3)
#             if goal_image.shape != (200, 200, 3):
#                 goal_pil = Image.fromarray(goal_image)
#                 goal_resized = goal_pil.resize((200, 200))
#                 goal_image = np.array(goal_resized)

#             goals.append(goal_image)

#         except Exception as e:
#             print(f"    Warning: Error generating goal for frame {i+1}: {e}")
#             # Use the last frame as fallback
#             if i > 0:
#                 goals.append(goals[-1])  # Use previous goal
#             else:
#                 goals.append(obs_image)  # Use observation as fallback

#     return np.array(goals, dtype=np.uint8)  # Shape: (seq_len, 200, 200, 3)

# def generate_susie_goals(image_sequence, language_instruction):
#     img_list = []
#     for img in image_sequence:
#         susie_img = susie_model.generate(language_instruction, img)
#         img_list.append(susie_img)
#     return np.array(img_list, dtype=np.uint8)

# Fixed batch size for SUSIE to prevent JIT recompilation
# SUSIE_BATCH_SIZE = 65
SUSIE_BATCH_SIZE = 128


def generate_susie_goals(image_sequence, language_instruction):
    """Generate SUSIE goals with fixed batch size to prevent recompilation."""
    seq_len = len(image_sequence)

    # If sequence fits in one batch, pad to fixed size
    if seq_len <= SUSIE_BATCH_SIZE:
        # Pad the sequence to SUSIE_BATCH_SIZE
        padded_sequence = np.zeros(
            (SUSIE_BATCH_SIZE, *image_sequence.shape[1:]), dtype=image_sequence.dtype
        )
        padded_sequence[:seq_len] = image_sequence

        # Generate goals for padded batch
        timer.tick("susie_goal_gen_single_batch")
        padded_goals = susie_model.generate_batch(language_instruction, padded_sequence)
        timer.tock("susie_goal_gen_single_batch")
        print(timer.get_total_times())

        # Return only the actual goals (remove padding)
        return padded_goals[:seq_len]
    else:
        # Process in multiple fixed-size batches
        goals = []
        for i in range(0, seq_len, SUSIE_BATCH_SIZE):
            batch_end = min(i + SUSIE_BATCH_SIZE, seq_len)
            batch = image_sequence[i:batch_end]

            # Pad the last batch if needed
            if len(batch) < SUSIE_BATCH_SIZE:
                padded_batch = np.zeros(
                    (SUSIE_BATCH_SIZE, *batch.shape[1:]), dtype=batch.dtype
                )
                padded_batch[: len(batch)] = batch
                batch_goals = susie_model.generate_batch(
                    language_instruction, padded_batch
                )
                goals.append(batch_goals[: len(batch)])
            else:
                batch_goals = susie_model.generate_batch(language_instruction, batch)
                goals.append(batch_goals)

        return np.concatenate(goals, axis=0)


def process_trajectory(function_data):
    global raw_dataset_path, tfrecord_dataset_path
    idx_range, letter, ctr, split, lang_ann = function_data
    unique_pid = split + "_" + letter + "_" + str(ctr)

    start_id, end_id = idx_range[0], idx_range[1]

    # We will filter the keys to only include what we need
    # Namely "rel_actions", "robot_obs", and "rgb_static"
    traj_rel_actions, traj_robot_obs, traj_rgb_static = [], [], []
    timer.tick("gcs_load_npz")
    for ep_id in range(start_id, end_id + 1):  # end_id is inclusive
        # print(unique_pid + ": iter " + str(ep_id-start_id) + " of " + str(end_id-start_id))

        ep_id = make_seven_characters(ep_id)

        # Load from GCS

        episode_path = os.path.join(
            raw_dataset_path, split, "episode_" + ep_id + ".npz"
        )
        timestep_data = gcs_load_npz(episode_path)

        rel_actions = timestep_data["rel_actions"]
        traj_rel_actions.append(rel_actions)

        robot_obs = timestep_data["robot_obs"]
        traj_robot_obs.append(robot_obs)

        rgb_static = timestep_data[
            "rgb_static"
        ]  # not normalized, so we have to do normalization in another script
        traj_rgb_static.append(rgb_static)
    timer.tock("gcs_load_npz")
    print(timer.get_total_times())

    timer.tick("np_array_convert")
    traj_rel_actions, traj_robot_obs, traj_rgb_static = (
        np.array(traj_rel_actions, dtype=np.float32),
        np.array(traj_robot_obs, dtype=np.float32),
        np.array(traj_rgb_static, dtype=np.uint8),
    )
    timer.tock("np_array_convert")

    # Generate SUSIE goal images
    timer.tick("susie_goal_gen")
    susie_goal_images = generate_susie_goals(traj_rgb_static, lang_ann)
    timer.tock("susie_goal_gen")
    print(timer.get_total_times())

    # Determine the output path in GCS
    timer.tick("tfrecord_write")
    write_dir = os.path.join(tfrecord_dataset_path, split, letter)

    # Write the TFRecord to GCS
    output_tfrecord_path = os.path.join(write_dir, "traj" + str(ctr) + ".tfrecord")

    with tf.io.TFRecordWriter(output_tfrecord_path) as writer:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "actions": tensor_feature(traj_rel_actions),
                    "proprioceptive_states": tensor_feature(traj_robot_obs),
                    "image_states": tensor_feature(traj_rgb_static),
                    "language_annotation": string_to_feature(lang_ann),
                    "susie_goal_images": tensor_feature(
                        susie_goal_images
                    ),  # Add SUSIE goals
                }
            )
        )
        writer.write(example.SerializeToString())
    timer.tock("tfrecord_write")
    print(timer.get_total_times())


# Let's prepare the inputs
function_inputs = []

# First let's do the train data
auto_lang_ann_path = os.path.join(
    raw_dataset_path, "training", "lang_annotations", "auto_lang_ann.npy"
)
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
    print(scene_info.keys())
    if "calvin_scene_D" in scene_info and start_idx <= scene_info["calvin_scene_D"][1]:
        ctr = D_ctr
        D_ctr += 1
        letter = "D"
    elif (
        start_idx <= scene_info["calvin_scene_B"][1]
    ):  # This is actually correct. In ascending order we have D, B, C, A
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

    function_inputs.append(
        (idx_range, letter, ctr, "training", all_language_annotations[i])
    )

print("Loading validation data annotations...")
# Next let's do the validation data
auto_lang_ann_path = os.path.join(
    raw_dataset_path, "validation", "lang_annotations", "auto_lang_ann.npy"
)
auto_lang_ann = gcs_load_npy(auto_lang_ann_path, allow_pickle=True)
auto_lang_ann = auto_lang_ann.item()
all_language_annotations = auto_lang_ann["language"]["ann"]
idx_ranges = auto_lang_ann["info"]["indx"]

print("Processing validation trajectories...")
ctr = 0
for i, idx_range in enumerate(idx_ranges):
    function_inputs.append(
        (idx_range, "D", ctr, "validation", all_language_annotations[i])
    )
    ctr += 1


# Finally loop through and process everything
print("Starting trajectory processing...")
processed_files = []
for function_input in tqdm(function_inputs, desc="Processing trajectories"):
    try:
        process_trajectory(function_input)
        # Keep track of processed files for verification
        idx_range, letter, ctr, split, lang_ann = function_input
        output_file = os.path.join(
            tfrecord_dataset_path, split, letter, f"traj{ctr}.tfrecord"
        )
        processed_files.append(output_file)
    except Exception as e:
        print(f"Error processing trajectory {function_input}: {e}")

print("Processing complete!")
