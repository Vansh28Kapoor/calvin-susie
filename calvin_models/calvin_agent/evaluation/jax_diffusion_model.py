from susie.model import create_sample_fn, new_create_sample_fn
from susie.jax_utils import initialize_compilation_cache
import numpy as np
from PIL import Image
import os
import jax
import jax.numpy as jnp
from typing import Union, List

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

class DiffusionModel:
    def __init__(self):
        # Check available hardware and optimize accordingly
        initialize_compilation_cache()

        self.sample_fn = create_sample_fn(
            os.getenv("DIFFUSION_MODEL_CHECKPOINT"),
            "kvablack/dlimp-diffusion/9n9ped8m",
            num_timesteps=200,
            prompt_w=7.5,
            context_w=1.5,
            eta=0.0,
            pretrained_path="runwayml/stable-diffusion-v1-5",
        )
        self.batch_sample_fn = new_create_sample_fn(
            os.getenv("DIFFUSION_MODEL_CHECKPOINT"),
            "kvablack/dlimp-diffusion/9n9ped8m",
            num_timesteps=200,
            prompt_w=7.5,
            context_w=1.5,
            eta=0.0,
            pretrained_path="runwayml/stable-diffusion-v1-5",
        )
    
    @staticmethod
    def _resize_to_256_jax(images):
        """JAX-accelerated resize to 256x256 using bilinear interpolation."""
        # images: (B, H, W, 3) or (H, W, 3)
        if images.ndim == 3:
            images = images[None, ...]
        
        B, H, W, C = images.shape
        # Use JAX's image resize function
        return jax.image.resize(images, (B, 256, 256, C), method='bilinear')
    
    @staticmethod 
    def _resize_to_200_jax(images):
        """JAX-accelerated resize to 200x200 using bilinear interpolation."""
        # images: (B, H, W, 3) or (H, W, 3)
        if images.ndim == 3:
            images = images[None, ...]
            
        B, H, W, C = images.shape
        # Use JAX's image resize function
        return jax.image.resize(images, (B, 200, 200, C), method='bilinear')
    
    def _resize_batch_pil_optimized(self, images: np.ndarray, target_size: tuple) -> np.ndarray:
        """Optimized PIL-based batch resizing with pre-allocation."""
        if images.ndim == 3:
            images = images[None, ...]
        
        B, H, W, C = images.shape
        # Pre-allocate output array for better memory efficiency
        resized = np.empty((B, target_size[1], target_size[0], C), dtype=images.dtype)
        
        # Use LANCZOS for better quality when downsampling
        resample_method = Image.LANCZOS if target_size[0] < W or target_size[1] < H else Image.BILINEAR
        
        for i in range(B):
            img_pil = Image.fromarray(images[i])
            resized_pil = img_pil.resize(target_size, resample_method)
            resized[i] = np.array(resized_pil)
        
        return resized
    def generate(self, language_command: str, image_obs: np.ndarray):
        """Single image generation (optimized with PIL)."""
        # Resize image to 256x256 using PIL
        image_256 = np.array(Image.fromarray(image_obs).resize((256, 256), Image.BILINEAR))
        image_256 = image_256.astype(np.uint8)
        
        sample = self.sample_fn(image_256, language_command, prompt_w=7.5, context_w=1.5)
        
        # Resize output to 200x200 using PIL
        goal_200 = np.array(Image.fromarray(sample).resize((200, 200), Image.BILINEAR))
        
        return goal_200.astype(np.uint8)

    # def generate_batch(self, language_command: str, batch_image_obs: np.ndarray):
    #     """
    #     Generate a batch of goal images from a batch of image observations and a language command,
    #     optimized with JAX's jit and vmap for resize operations.
    #     """
    #     # Ensure input is a JAX array
    #     batch_image_obs_jax = jnp.asarray(batch_image_obs)
        
    #     # JIT-compiled batch resize to 256x256
    #     @jax.jit
    #     def _batch_resize_256(images):
    #         return jax.vmap(lambda img: jax.image.resize(img, (256, 256, 3), method='bilinear'))(images)
        
    #     # JIT-compiled batch resize to 200x200
    #     @jax.jit
    #     def _batch_resize_200(images):
    #         return jax.vmap(lambda img: jax.image.resize(img, (200, 200, 3), method='bilinear'))(images)
        
    #     # Resize inputs to 256x256 using JIT+vmap
    #     batch_256_jax = _batch_resize_256(batch_image_obs_jax)
    #     batch_256 = np.asarray(batch_256_jax).astype(np.uint8)
        
    #     # Run the batch sampling function (this handles batching internally)
    #     samples_batch = self.batch_sample_fn(batch_256, language_command)
        
    #     # Resize outputs to 200x200 using JIT+vmap
    #     samples_jax = jnp.asarray(samples_batch)
    #     batch_200_jax = _batch_resize_200(samples_jax)
    #     batch_goals = np.asarray(batch_200_jax).astype(np.uint8)
        
    #     return batch_goals
    def generate_batch(self, language_command: str, batch_image_obs: np.ndarray):
        timer.tick("host to Device")
        x = jnp.asarray(batch_image_obs)           # H2D once
        timer.tock("host to Device")
        timer.tick("JAX resize_to_256_u8")
        x256 = _resize_to_256_u8(x)                # on device
        timer.tock("JAX resize_to_256_u8")
        timer.tick("JAX Batch sample")
        y256 = self.batch_sample_fn(x256, language_command)  # stays on device
        timer.tock("JAX Batch sample")
        timer.tick("JAX resize_to_200_u8")
        y200 = _resize_to_200_u8(y256)             # on device
        timer.tock("JAX resize_to_200_u8")
        timer.tick("Device to host")
        output = np.asarray(y200)
        timer.tock("Device to host")
        print(timer.get_total_times())
        return output
    
# top-level (module scope), compile once
@jax.jit
def _resize_to_256_u8(batch_u8):
    # jax.image.resize expects float; keep on device
    b = batch_u8.shape[0]
    x = batch_u8.astype(jnp.float32)
    x = jax.image.resize(x, (b, 256, 256, 3), method="bilinear")
    return jnp.clip(jnp.round(x), 0, 255).astype(jnp.uint8)

@jax.jit
def _resize_to_200_u8(batch_u8):
    b = batch_u8.shape[0]
    x = batch_u8.astype(jnp.float32)
    x = jax.image.resize(x, (b, 200, 200, 3), method="bilinear")
    return jnp.clip(jnp.round(x), 0, 255).astype(jnp.uint8)