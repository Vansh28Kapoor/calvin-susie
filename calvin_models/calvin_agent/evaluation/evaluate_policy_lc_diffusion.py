import argparse
from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import sys
import time
import requests
import json

# Set headless backend before importing cv2 to avoid Qt display issues
# Disable all GUI backends and Qt entirely
os.environ['MPLBACKEND'] = 'Agg'
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Use X11 backend which is available
os.environ['DISPLAY'] = ''  # But don't actually connect to display
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_INTEL_MFX'] = '0'

# Set JAX flags to avoid compilation cache issues
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1 --xla_gpu_enable_triton_softmax_fusion=false --xla_gpu_triton_gemm_any=false'
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cache'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Don't import cv2 at all to avoid Qt issues
# import cv2
from tqdm import tqdm

# Monkey patch for numpy compatibility with older dependencies
import numpy as np
if not hasattr(np, 'int'):
    np.int = np.int64
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'bool'):
    np.bool = np.bool_

import jax_diffusion_model  # For subgoal generation
import lc_diffusion_policy


# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm
from PIL import Image

from calvin_env.envs.play_table_env import get_env

logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = int(os.getenv("NUM_EVAL_SEQUENCES"))

def safe_create_video(image_list, output_path, size=(200, 200), fps=15):
    """Create video using imageio to avoid Qt issues in headless environments."""
    try:
        import imageio
        
        # Ensure size is divisible by 16 for video compatibility
        adjusted_size = (
            ((size[0] + 15) // 16) * 16,  # Round up to nearest multiple of 16
            ((size[1] + 15) // 16) * 16
        )
        
        # Save individual frames first as backup
        print(f"Creating video with {len(image_list)} frames: {output_path}")
        img_dir = output_path.replace('.mp4', '_frames')
        os.makedirs(img_dir, exist_ok=True)
        
        # Prepare images for video creation
        processed_images = []
        for i, img in enumerate(image_list):
            # Convert numpy array to PIL Image for resizing
            pil_img = Image.fromarray(img.astype(np.uint8))
            # Resize to video-compatible size
            pil_img = pil_img.resize(adjusted_size, Image.LANCZOS)
            
            # Convert back to numpy array
            processed_img = np.array(pil_img)
            processed_images.append(processed_img)
            
            # Also save individual frame as backup (at original size)
            img_path = os.path.join(img_dir, f"frame_{i:04d}.png")
            original_pil = Image.fromarray(img.astype(np.uint8))
            if original_pil.size != size:
                original_pil = original_pil.resize(size, Image.LANCZOS)
            original_pil.save(img_path)
        
        # Create video using imageio
        try:
            with imageio.get_writer(output_path, fps=fps, codec='libx264', pixelformat='yuv420p') as writer:
                for img in processed_images:
                    writer.append_data(img)
            
            return True
            
        except Exception as video_error:
            print(f"Video creation failed ({video_error}), but frames saved to {img_dir}")
            return True  # Still return True since we have the frames
        
    except ImportError:
        print("imageio not available, saving frames only...")
        # Fallback to frame-only saving
        img_dir = output_path.replace('.mp4', '_frames')
        os.makedirs(img_dir, exist_ok=True)
        
        for i, img in enumerate(image_list):
            img_path = os.path.join(img_dir, f"frame_{i:04d}.png")
            pil_img = Image.fromarray(img.astype(np.uint8))
            if pil_img.size != size:
                pil_img = pil_img.resize(size, Image.LANCZOS)
            pil_img.save(img_path)
        
        return True
        
    except Exception as e:
        print(f"Error in safe_create_video {output_path}: {e}")
        return False

def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


class LanguageConditionedModel(CalvinBaseModel):
    """Custom model that uses language-conditioned diffusion policy with SUSIE subgoal generation.
    
    This model:
    1. Uses diffusion model to generate visual subgoals from language instructions
    2. Concatenates subgoals with observations (matching training data format)
    3. Uses lc_ddpm_bc agent trained with load_susie_goal_images=True
    """
    
    def __init__(self):
        self.diffusion_model = jax_diffusion_model.DiffusionModel()
        self.lc_policy = lc_diffusion_policy.LCDiffusionPolicy()
        self.log_dir = "experiments_lc"
        self.episode_counter = None
        self.language_task = None
        self.obs_image_seq = None
        self.goal_image_seq = None
        self.action_seq = None
        self.combined_images = None

        # Other necessary variables for running rollouts
        self.goal_image = None
        self.subgoal_counter = 0
        self.subgoal_max = 5  # Regenerate subgoal every 5 steps
        self.pbar = None

    def reset(self):
        if self.episode_counter is None: # this is the first time reset has been called
            self.episode_counter = 0
            self.obs_image_seq = []
            self.goal_image_seq = []
            self.action_seq = []
            self.combined_images = []
        else:
            episode_log_dir = os.path.join(self.log_dir, "ep" + str(self.episode_counter))
            if not os.path.exists(episode_log_dir):
                os.makedirs(episode_log_dir)

            # Log the language task
            with open(os.path.join(episode_log_dir, "language_task.txt"), "w") as f:
                f.write(self.language_task)
            
            # Log the observation video
            safe_create_video(
                self.obs_image_seq, 
                os.path.join(episode_log_dir, "trajectory.mp4"),
                size=(200, 200),
                fps=15
            )

            # Log the goals video
            safe_create_video(
                self.goal_image_seq,
                os.path.join(episode_log_dir, "goals.mp4"),
                size=(200, 200), 
                fps=15
            )

            # Log the combined image
            safe_create_video(
                self.combined_images,
                os.path.join(episode_log_dir, "combined.mp4"),
                size=(400, 200),
                fps=15
            )

            # Log the actions
            np.save(os.path.join(episode_log_dir, "actions.npy"), np.array(self.action_seq))

            # Update/reset all the variables
            self.episode_counter += 1
            self.obs_image_seq = []
            self.goal_image_seq = []
            self.action_seq = []
            self.goal_image = None
            self.combined_images = []
            self.subgoal_counter = 0

            # Reset the LC policy
            self.lc_policy.reset()

        # tqdm progress bar
        if self.pbar is not None:
            self.pbar.close()
        self.pbar = tqdm(total=EP_LEN)

    def step(self, obs, goal):
        """
        Args:
            obs: environment observations
            goal: language instruction (string)
        Returns:
            action: predicted action
        """
        rgb_obs = obs["rgb_obs"]["rgb_static"]
        self.language_task = goal

        # If we need to, generate a new goal image using the diffusion model
        if self.goal_image is None or self.subgoal_counter >= self.subgoal_max:
            self.goal_image = self.diffusion_model.generate(self.language_task, rgb_obs)
            self.subgoal_counter = 0

        # Log the image observation and the goal image
        self.obs_image_seq.append(rgb_obs)
        self.goal_image_seq.append(self.goal_image)
        self.combined_images.append(np.concatenate([rgb_obs, self.goal_image], axis=1))
        assert self.combined_images[-1].shape == (200, 400, 3)

        # Query the language-conditioned behavior cloning model with concatenated obs+goal
        # The policy expects the goal image concatenated with obs (matching training format)
        action_cmd = self.lc_policy.predict_action(rgb_obs, self.goal_image, self.language_task)

        # Log the predicted action
        self.action_seq.append(action_cmd)

        # Update variables
        self.subgoal_counter += 1

        # Update progress bar
        self.pbar.update(1)

        return action_cmd


def evaluate_policy(model, env, epoch=0, eval_log_dir=None, debug=False, create_plan_tsne=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(__file__).absolute().parents[2] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES) #check

    results = []
    plans = defaultdict(list)

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)
    print_and_save(results, eval_sequences, eval_log_dir, epoch)

    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask in eval_sequence:
        success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()

    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    for step in range(EP_LEN):
        action = model.step(obs, lang_annotation)
        obs, _, _, current_info = env.step(action)
        if debug:
            # Skip rendering to avoid Qt issues in headless environments
            # img = env.render(mode="rgb_array")
            # join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
            pass
        if step == 0:
            # for tsne plot, only if available
            collect_plan(model, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
            return True
    if debug:
        print(colored("fail", "red"), end=" ")
    return False


def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a language-conditioned trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

    # arguments for loading default model
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path of the checkpoint",
    )
    parser.add_argument(
        "--last_k_checkpoints",
        type=int,
        help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    )

    # arguments for loading custom model or custom language embeddings
    parser.add_argument(
        "--custom_model", action="store_true", help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")

    parser.add_argument("--eval_log_dir", default=None, type=str, help="Where to log the evaluation results.")

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    # evaluate a custom model
    if args.custom_model:
        model = LanguageConditionedModel()
        env = make_env(args.dataset_path)
        evaluate_policy(model, env, debug=args.debug)
    else:
        assert "train_folder" in args

        checkpoints = []
        if args.checkpoints is None and args.last_k_checkpoints is None and args.checkpoint is None:
            print("Evaluating model with last checkpoint.")
            checkpoints = [get_last_checkpoint(Path(args.train_folder))]
        elif args.checkpoints is not None:
            print(f"Evaluating model with checkpoints {args.checkpoints}.")
            checkpoints = get_checkpoints_for_epochs(Path(args.train_folder), args.checkpoints)
        elif args.checkpoints is None and args.last_k_checkpoints is not None:
            print(f"Evaluating model with last {args.last_k_checkpoints} checkpoints.")
            checkpoints = get_all_checkpoints(Path(args.train_folder))[-args.last_k_checkpoints :]
        elif args.checkpoint is not None:
            checkpoints = [Path(args.checkpoint)]

        env = None
        for checkpoint in checkpoints:
            epoch = checkpoint.stem.split("=")[1]
            model, env, _ = get_default_model_and_env(
                args.train_folder,
                args.dataset_path,
                checkpoint,
                env=env,
                device_id=args.device,
            )
            evaluate_policy(model, env, epoch, eval_log_dir=args.eval_log_dir, debug=args.debug, create_plan_tsne=True)


if __name__ == "__main__":
    main()