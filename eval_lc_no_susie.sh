#!/bin/bash

# Evaluation script for Language-Conditioned Policy WITHOUT SUSIE subgoal generation
# This script evaluates the lc_ddpm_bc model trained WITHOUT load_susie_goal_images
# 
# Key components:
# 1. Policy works directly with language instructions and observations
# 2. No visual subgoal generation or concatenation
# 3. Uses lc_ddpm_bc agent trained with load_susie_goal_images=False

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

# Set checkpoint path for LC policy WITHOUT SUSIE
export LC_POLICY_CHECKPOINT_NO_SUSIE=new_lcbc_trained_checkpoints/jaxrl_m_calvin_lcbc/_20251025_095858
export NUM_EVAL_SEQUENCES=10

echo "Evaluating Language-Conditioned Policy WITHOUT SUSIE..."
echo "LC Policy Checkpoint (No SUSIE): $LC_POLICY_CHECKPOINT_NO_SUSIE"
echo "Number of evaluation sequences: $NUM_EVAL_SEQUENCES"

python calvin_models/calvin_agent/evaluation/evaluate_policy_lc_no_susie.py \
    --dataset_path mini_dataset \
    --custom_model \
    --debug