"""
VoxCPM2 Colab Kurulum Scripti
Kullanimi:  Colab'da yeni bir hucreye yapistir ve calistir.
Tek hucrede her seyi halleder.
"""
import subprocess, sys, os, json, time, asyncio, numpy as np
from IPython.display import Audio, display

print("=" * 50)
print("VoxCPM2 Colab Kurulum ve Inference")
print("=" * 50)

# 1. Torch
print("\n[1/5] Torch 2.5.1 kuruluyor...")
subprocess.run([sys.executable, "-m", "pip", "install",
    "torch==2.5.1", "torchaudio==2.5.1",
    "--index-url", "https://download.pytorch.org/whl/cu121", "-q"])

# 2. flash-attn
print("[2/5] flash-attn kuruluyor (3-4 dk)...")
subprocess.run([sys.executable, "-m", "pip", "install",
    "flash-attn", "--no-build-isolation", "--no-cache-dir", "-q"])

# 3. Diger bagimliliklar
print("[3/5] Diger paketler kuruluyor...")
subprocess.run([sys.executable, "-m", "pip", "install",
    "transformers>=4.51.0", "soundfile", "librosa", "numpy",
    "tqdm", "xxhash", "pydantic", "torchcodec",
    "huggingface_hub", "psutil", "-q"])

# 4. Kaynak kodu clone + kur
print("[4/5] nano-vllm-voxcpm kuruluyor...")
REPO = "https://github.com/kstbhdr/nanovllm-voxcpm-official"
WORK = "/content/nanovllm-voxcpm-official"
if not os.path.exists(WORK):
    subprocess.run(["git", "clone", REPO, WORK], capture_output=True)
os.chdir(WORK)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "-q"])

from nanovllm_voxcpm import VoxCPM
print("  VoxCPM import OK")

# 5. Model indir + inference
print("[5/5] Model indiriliyor ve inference calistiriliyor...")
from huggingface_hub import snapshot_download
model_path = snapshot_download(repo_id="openbmb/VoxCPM2")

server = VoxCPM.from_pretrained(
    model=model_path,
    max_num_batched_tokens=4096, max_num_seqs=2,
    max_model_len=2048, gpu_memory_utilization=0.85,
    enforce_eager=True, devices=[0],
)

async def run():
    await server.wait_for_ready()
    buf = []
    t0 = time.time()
    async for data in server.generate(
        target_text="Merhaba dunya! Bugun hava cok guzel.",
        cfg_value=2.0, temperature=1.0,
    ):
        buf.append(data)
    wav = np.concatenate(buf, axis=0)
    model_info = await server.get_model_info()
    sr = int(model_info["sample_rate"])
    print(f"\nSure: {time.time()-t0:.2f}s")
    print(f"Ses: {wav.shape[0]/sr:.2f}s")
    display(Audio(wav, rate=sr))
    import soundfile as sf
    sf.write("voxcpm2_output.wav", wav, sr)
    print("Kaydedildi: voxcpm2_output.wav")
    await server.stop()

await run()
print("\nBASARILI!")
