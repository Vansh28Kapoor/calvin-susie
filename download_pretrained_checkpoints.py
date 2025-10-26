#!/usr/bin/env python3
"""
Download SUSIE checkpoints from HuggingFace
"""

import os
from huggingface_hub import snapshot_download

def download_checkpoints():
    """Download the SUSIE checkpoints."""
    
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Download the entire repository first
    print("Downloading checkpoints repository...")
    repo_path = snapshot_download(
        repo_id="patreya/susie-calvin-checkpoints",
        local_dir="checkpoints",
        local_dir_use_symlinks=False
    )
    
    diffusion_path = os.path.join("checkpoints", "diffusion_model")
    gc_policy_path = os.path.join("checkpoints", "gc_policy")
    
    print(f"Repository downloaded to: {repo_path}")
    print(f"Diffusion model path: {diffusion_path}")
    print(f"GC policy path: {gc_policy_path}")
    
    return diffusion_path, gc_policy_path

if __name__ == "__main__":
    print("Downloading SUSIE checkpoints from HuggingFace...")
    diffusion_path, gc_path = download_checkpoints()
    print("Download complete!")
    print(f"Diffusion model: {diffusion_path}")
    print(f"GC Policy: {gc_path}")