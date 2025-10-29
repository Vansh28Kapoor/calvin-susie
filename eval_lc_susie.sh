#!/bin/bash

# Evaluation script for Language-Conditioned SUSIE Policy
# This script evaluates the lc_ddpm_bc model trained with load_susie_goal_images=True
# 
# Key components:
# 1. Diffusion model generates visual subgoals from language instructions
# 2. Subgoals are concatenated with observations (obs: 3ch + goal: 3ch = 6ch total)
# 3. lc_ddpm_bc policy is trained on this concatenated format

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate susie-calvin

# Fix CuDNN version mismatch
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# export TF_FORCE_GPU_ALLOW_GROWTH=true
# # Skip CuDNN version check
# export TF_CUDNN_VERSION_MISMATCH_OK=1

# Set environment variables for headless operation
export MPLBACKEND=Agg
export DISPLAY=""
export QT_QPA_PLATFORM=xcb
export OPENCV_VIDEOIO_PRIORITY_MSMF=0
export OPENCV_VIDEOIO_PRIORITY_INTEL_MFX=0

# Set JAX flags to avoid compilation cache issues
mkdir -p /tmp/jax_cache
export XLA_FLAGS="--xla_force_host_platform_device_count=1 --xla_gpu_enable_triton_softmax_fusion=false --xla_gpu_triton_gemm_any=false"
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Clear any existing cache to avoid corruption
rm -rf /tmp/jax_cache/*

export DIFFUSION_MODEL_CHECKPOINT=checkpoints/diffusion_model
export LC_POLICY_CHECKPOINT=susie_lcbc_trained_checkpoints/jaxrl_m_calvin_lcbc/_20251025_104227
export NUM_EVAL_SEQUENCES=10

echo "Evaluating Language-Conditioned SUSIE Policy..."
echo "Diffusion Model Checkpoint: $DIFFUSION_MODEL_CHECKPOINT"
echo "LC Policy Checkpoint: $LC_POLICY_CHECKPOINT"
echo "Number of evaluation sequences: $NUM_EVAL_SEQUENCES"

python calvin_models/calvin_agent/evaluation/evaluate_policy_lc_diffusion.py \
    --dataset_path mini_dataset \
    --custom_model \
    --debug