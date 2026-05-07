"""
###############################################################################
# VoxCPM2 - COMPLETE SETUP + INFERENCE (Colab / WSL / Linux)
# 
# Kullanim:
#   Colab:  Bu dosyayi Colab'a yukle, calistir
#   WSL:    python3 voxcpm2_universal.py
#
# Ne yapar:
#   1. torch 2.5.1 + flash-attn kurar
#   2. nano-vllm-voxcpm (VRAM fix'li fork) kurar
#   3. openbmb/VoxCPM2 modelini indirir
#   4. "Merhaba dunya!" sesini sentezler
#   5. Ciktiyi voxcpm2_output.wav olarak kaydeder
###############################################################################
"""

import subprocess, sys, os, time, asyncio, numpy as np

PY = sys.executable

def run(cmd, desc):
    print(f"[{desc}]...", end=" ", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        print("OK")
    else:
        print(f"HATA: {r.stderr[-200:]}")
    return r.returncode

# 1. Torch
run([PY, "-m", "pip", "install", "torch==2.5.1", "torchaudio==2.5.1",
     "--index-url", "https://download.pytorch.org/whl/cu121", "-q"], "Torch 2.5.1")

# 2. flash-attn
run([PY, "-m", "pip", "install", "flash-attn", "--no-build-isolation", "-q"], "flash-attn")

# 3. Diger paketler
run([PY, "-m", "pip", "install", "transformers>=4.51.0", "soundfile", "librosa",
     "numpy", "tqdm", "xxhash", "pydantic", "torchcodec", "huggingface_hub",
     "psutil", "-q"], "Diger paketler")

# 4. Fork'u kur (VRAM fix dahil)
REPO = "https://github.com/kstbhdr/nanovllm-voxcpm-official"
WORK = "/content/nanovllm-voxcpm-official" if "colab" in sys.modules else "/tmp/nanovllm-voxcpm"
if not os.path.exists(WORK):
    subprocess.run(["git", "clone", REPO, WORK, "--depth=1"], capture_output=True)
os.chdir(WORK)
run([PY, "-m", "pip", "install", "-e", ".", "--no-deps", "-q"], "nano-vllm-voxcpm")

from nanovllm_voxcpm import VoxCPM

# 5. Modeli indir + inference
print("\n[Model indiriliyor + Inference]...")
from huggingface_hub import snapshot_download
model_path = snapshot_download(repo_id="openbmb/VoxCPM2")

async def infer():
    s = VoxCPM.from_pretrained(model=model_path, max_num_batched_tokens=4096,
        max_num_seqs=2, max_model_len=2048, gpu_memory_utilization=0.85,
        enforce_eager=True, devices=[0])
    await s.wait_for_ready()
    buf, t0 = [], time.time()
    async for d in s.generate(target_text="Merhaba dunya! Bugun hava cok guzel.", cfg_value=2.0):
        buf.append(d)
    wav = np.concatenate(buf)
    sr = int((await s.get_model_info())["sample_rate"])
    print(f"  Sure: {time.time()-t0:.2f}s | Ses: {wav.shape[0]/sr:.2f}s")
    import soundfile as sf
    sf.write("voxcpm2_output.wav", wav, sr)
    print(f"  Kaydedildi: voxcpm2_output.wav")
    await s.stop()

asyncio.run(infer())
print("\nBASARILI!")
