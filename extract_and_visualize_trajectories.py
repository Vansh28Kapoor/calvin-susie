import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.python.lib.io import file_io
from tqdm import tqdm
import sys

# --- Configuration ---
GCS_BASE_PATH = "gs://max-us-central2/susieprocessed_tfrecords/training"
LOCAL_OUTPUT_DIR = "trajectory_visualizations"
COLS_PER_ROW = 5  # Number of timesteps to show per row in the visualization

# --- TFRecord Parsing Function ---
def parse_tfrecord(example_proto):
    """Parse a single TFRecord example."""
    feature_description = {
        'actions': tf.io.FixedLenFeature([], tf.string),
        'proprioceptive_states': tf.io.FixedLenFeature([], tf.string),
        'image_states': tf.io.FixedLenFeature([], tf.string),
        'language_annotation': tf.io.FixedLenFeature([], tf.string),
        'susie_goal_images': tf.io.FixedLenFeature([], tf.string),
    }
    
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    # Deserialize tensors
    actions = tf.io.parse_tensor(parsed_features['actions'], out_type=tf.float32)
    proprioceptive_states = tf.io.parse_tensor(parsed_features['proprioceptive_states'], out_type=tf.float32)
    image_states = tf.io.parse_tensor(parsed_features['image_states'], out_type=tf.uint8)
    language_annotation = parsed_features['language_annotation']
    susie_goal_images = tf.io.parse_tensor(parsed_features['susie_goal_images'], out_type=tf.uint8)
    
    return {
        'actions': actions,
        'proprioceptive_states': proprioceptive_states,
        'image_states': image_states,
        'language_annotation': language_annotation,
        'susie_goal_images': susie_goal_images
    }

# --- Visualization Function ---
def create_and_save_visualization(image_states, susie_goal_images, language_annotation, output_path):
    """Generates and saves a visualization of the trajectory."""
    traj_len = len(image_states)
    if traj_len == 0:
        return

    # Calculate grid dimensions
    rows = (traj_len + COLS_PER_ROW - 1) // COLS_PER_ROW

    fig, axes = plt.subplots(rows, COLS_PER_ROW * 2, figsize=(24, 3 * rows))
    fig.suptitle(f"Observation vs SUSIE Goal - '{language_annotation}'", fontsize=16, y=0.995)

    for i in range(traj_len):
        row = i // COLS_PER_ROW
        col = i % COLS_PER_ROW
        
        # Ensure axes is a 2D array for consistent indexing
        if rows == 1:
            ax_obs = axes[col * 2]
            ax_goal = axes[col * 2 + 1]
        else:
            ax_obs = axes[row, col * 2]
            ax_goal = axes[row, col * 2 + 1]

        # Observation image (left)
        ax_obs.imshow(image_states[i])
        ax_obs.set_title(f'Obs t={i}', fontsize=9, fontweight='bold', color='blue')
        ax_obs.axis('off')
        for spine in ax_obs.spines.values():
            spine.set_edgecolor('blue')
            spine.set_linewidth(2)
            spine.set_visible(True)
        
        # SUSIE goal image (right)
        ax_goal.imshow(susie_goal_images[i])
        ax_goal.set_title(f'Goal t={i}', fontsize=9, fontweight='bold', color='red')
        ax_goal.axis('off')
        for spine in ax_goal.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2)
            spine.set_visible(True)

    # Hide unused subplots
    total_subplots = rows * COLS_PER_ROW
    for i in range(traj_len, total_subplots):
        row = i // COLS_PER_ROW
        col = i % COLS_PER_ROW
        if rows == 1:
            axes[col * 2].axis('off')
            axes[col * 2 + 1].axis('off')
        else:
            axes[row, col * 2].axis('off')
            axes[row, col * 2 + 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path)
    plt.close(fig)  # Close the figure to free memory

# --- Main Logic ---
def main():
    """Main function to process all trajectories."""
    print(f"Starting trajectory visualization process.")
    print(f"GCS Source: {GCS_BASE_PATH}")
    print(f"Local Destination: {LOCAL_OUTPUT_DIR}")

    # Create the main local output directory
    if not os.path.exists(LOCAL_OUTPUT_DIR):
        os.makedirs(LOCAL_OUTPUT_DIR)

    try:
        # List subdirectories (A, B, C, D, etc.)
        subdirs = [os.path.basename(f.rstrip('/')) for f in file_io.list_directory(GCS_BASE_PATH)]
        print(f"Found subdirectories: {subdirs}")
    except Exception as e:
        print(f"Error listing GCS directories: {e}")
        print("Please ensure you have authenticated with GCS.")
        return

    for subdir in subdirs:
        gcs_subdir_path = os.path.join(GCS_BASE_PATH, subdir)
        local_subdir_path = os.path.join(LOCAL_OUTPUT_DIR, subdir)

        # Create local subdirectory
        if not os.path.exists(local_subdir_path):
            os.makedirs(local_subdir_path)
        
        print(f"\nProcessing subdirectory: {subdir}")

        try:
            # Get all tfrecord files in the GCS subdirectory
            tfrecord_files = file_io.get_matching_files(os.path.join(gcs_subdir_path, "*.tfrecord"))
            print(f"Found {len(tfrecord_files)} trajectories.")

            if not tfrecord_files:
                continue

            # Process each trajectory file
            for tfrecord_path in tqdm(tfrecord_files, desc=f"Visualizing {subdir}", unit="traj"):
                try:
                    # Load the dataset
                    dataset = tf.data.TFRecordDataset(tfrecord_path)
                    dataset = dataset.map(parse_tfrecord)
                    
                    # Get the trajectory data
                    trajectory = next(iter(dataset))
                    
                    # Extract data
                    image_states = trajectory['image_states'].numpy()
                    susie_goal_images = trajectory['susie_goal_images'].numpy()
                    language_annotation = trajectory['language_annotation'].numpy().decode('utf-8')
                    
                    # Define output path
                    traj_filename = os.path.basename(tfrecord_path).replace('.tfrecord', '.png')
                    output_image_path = os.path.join(local_subdir_path, traj_filename)
                    
                    # Create and save the visualization
                    create_and_save_visualization(
                        image_states, 
                        susie_goal_images, 
                        language_annotation, 
                        output_image_path
                    )
                except Exception as e:
                    print(f"  Error processing file {tfrecord_path}: {e}")
                    continue

        except Exception as e:
            print(f"Error accessing files in {gcs_subdir_path}: {e}")
            continue
            
    print("\nVisualization process complete!")

if __name__ == "__main__":
    # Suppress TensorFlow warnings for cleaner output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    main()
