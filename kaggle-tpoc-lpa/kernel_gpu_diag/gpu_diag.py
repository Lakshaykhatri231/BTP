# %%
import os, subprocess, sys, glob
def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()

print("=== nvidia-smi ===", flush=True)
print(sh("nvidia-smi 2>&1 | head -20") or "(no output)", flush=True)
print("\n=== /proc/driver/nvidia ===", flush=True)
print(sh("ls /proc/driver/nvidia 2>&1 | head"), flush=True)
print("\n=== /dev/nvidia* ===", flush=True)
print(sh("ls -la /dev/nvidia* 2>&1 | head"), flush=True)
print("\n=== torch ===", flush=True)
try:
    import torch
    print("torch", torch.__version__, "| built with cuda:", torch.version.cuda,
          "| is_available:", torch.cuda.is_available(),
          "| device_count:", torch.cuda.device_count(), flush=True)
    if torch.cuda.is_available():
        print("device 0:", torch.cuda.get_device_name(0),
              "| mem GB:", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), flush=True)
except Exception as e:
    print("torch import failed:", type(e).__name__, e, flush=True)
print("\n=== env hints ===", flush=True)
for k in sorted(os.environ):
    if any(t in k.upper() for t in ("CUDA", "NVIDIA", "GPU", "ACCEL", "KAGGLE_KERNEL", "KAGGLE_DOCKER", "TPU")):
        print(f"  {k}={os.environ[k][:100]}", flush=True)
print("\n=== cpu/mem ===", flush=True)
print(sh("nproc; free -g | head -2"), flush=True)
