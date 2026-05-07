"""WSL import test"""
import torch
print(f"torch: {torch.__version__}")
print(f"cuda: {torch.cuda.is_available()}")
print(f"gpu: {torch.cuda.get_device_name(0)}")

import flash_attn
print("flash-attn: OK")

from nanovllm_voxcpm import VoxCPM
print("VoxCPM import: OK")
print("\nAll imports successful!")
