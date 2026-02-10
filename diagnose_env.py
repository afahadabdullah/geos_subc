import torch
import sys
import os
import socket

print(f"Hostname: {socket.gethostname()}")
print(f"Python: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
else:
    print("WARNING: CUDA is NOT available to Torch.")

try:
    import accelerate
    print(f"Accelerate version: {accelerate.__version__}")
except ImportError:
    print("Accelerate NOT installed.")

# Check for large memory usage
import gc
print("Memory Check (GC):")
gc.collect()
print(f"GC Objects: {len(gc.get_objects())}")
