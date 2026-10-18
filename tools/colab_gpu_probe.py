# Probe a Colab VM: GPU, torch, disk, RAM, and whether the hosts VAR-Lab M4 needs are reachable.
# Run from the Mac:  colab run --gpu L4 --timeout 120 -s varlab-probe tools/colab_gpu_probe.py
# (colab run provisions a fresh VM, runs this, then stops the VM; nothing is left running.)
import os, shutil, subprocess, sys, urllib.request

print("python", sys.version.split()[0])
smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True)
print("gpu", smi.stdout.strip() or "nvidia-smi: no GPU")
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("device", p.name, round(p.total_memory / 2**30, 1), "GiB")
except Exception as e:  # noqa: BLE001
    print("torch import failed:", e)
root = "/content" if os.path.isdir("/content") else "/"
t, _, f = shutil.disk_usage(root)
print("disk free GiB", round(f / 2**30, 1), "of", round(t / 2**30, 1), "at", root)
with open("/proc/meminfo") as fh:
    print("ram", fh.readline().strip())
print("cpus", os.cpu_count())
for host in ["https://huggingface.co", "https://www.soccer-net.org", "https://pypi.org", "https://github.com"]:
    try:
        r = urllib.request.urlopen(urllib.request.Request(host, method="HEAD"), timeout=10)
        print("reach", host, r.status)
    except Exception as e:  # noqa: BLE001
        print("reach", host, "FAIL", type(e).__name__)
