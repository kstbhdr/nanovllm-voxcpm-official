#!/usr/bin/env python3
"""VoxCPM2 Colab Tek Tıkla Kurulum + Inference
Kullanım: Colab'da yeni bir kod hücresine kopyala-yapıştır, çalıştır.
"""
import subprocess, sys, os, json, time, asyncio, numpy as np
from IPython.display import Audio, display
from huggingface_hub import snapshot_download

print("="*60)
print("VoxCPM2 Colab Kurulum ve Inference")
print("="*60)

# 1. Torch
print("\n[1/4] Torch 2.5.1 kuruluyor...")
subprocess.run([sys.executable, "-m", "pip", "install",
    "torch==2.5.1", "torchaudio==2.5.1",
    "--index-url", "https://download.pytorch.org/whl/cu121", "-q"])

# 2. flash-attn
print("[2/4] flash-attn kuruluyor (~3 dk)...")
subprocess.run([sys.executable, "-m", "pip", "install",
    "flash-attn", "--no-build-isolation", "--no-cache-dir", "-q"])

# 3. Diger paketler + nano-vllm-voxcpm
print("[3/4] Diger paketler ve nano-vllm-voxcpm kuruluyor...")
subprocess.run([sys.executable, "-m", "pip", "install",
    "transformers>=4.51.0", "soundfile", "librosa", "numpy",
    "tqdm", "xxhash", "pydantic", "torchcodec",
    "huggingface_hub", "psutil", "-q"])

REPO = "https://github.com/kstbhdr/nanovllm-voxcpm-official"
WORK = "/content/nanovllm-voxcpm-official"
if not os.path.exists(WORK):
    subprocess.run(["git", "clone", REPO, WORK], capture_output=True)
os.chdir(WORK)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "-q"])
from nanovllm_voxcpm import VoxCPM
print("  VoxCPM import OK")

# 4. Model indir + Inference
print("[4/4] openbmb/VoxCPM2 indiriliyor ve inference calistiriliyor...")
model_path = snapshot_download(repo_id="openbmb/VoxCPM2")

server = VoxCPM.from_pretrained(
    model=model_path,
    max_num_batched_tokens=4096, max_num_seqs=2,
    max_model_len=2048, gpu_memory_utilization=0.85,
    enforce_eager=True, devices=[0],
)

async def run():
    await server.wait_for_ready()
    text = "Merhaba dunya! Bugun hava cok guzel."
    buf = []
    t0 = time.time()
    async for data in server.generate(target_text=text, cfg_value=2.0, temperature=1.0):
        buf.append(data)
    wav = np.concatenate(buf, axis=0)
    mi = await server.get_model_info()
    sr = int(mi["sample_rate"])
    print(f"\nSure: {time.time()-t0:.2f}s | Ses: {wav.shape[0]/sr:.2f}s")
    display(Audio(wav, rate=sr))
    import soundfile as sf
    sf.write("voxcpm2_output.wav", wav, sr)
    print("Kaydedildi: voxcpm2_output.wav")
    await server.stop()

await run()
print("\nBASARIYLA TAMAMLANDI!")
