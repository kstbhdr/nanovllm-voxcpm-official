"""
VoxCPM2 Inference - Standalone Script
Colab'da calistirmak icin:  python voxcpm2_run.py
"""

import subprocess
import sys
import importlib
import os

REQUIRED_PKGS = [
    ("torch", "torch>=2.5.1"),
    ("flash_attn", "flash-attn"),
    ("triton", "triton>=3.0.0"),
    ("transformers", "transformers>=4.51.0"),
    ("soundfile", "soundfile"),
    ("librosa", "librosa"),
    ("numpy", "numpy"),
    ("tqdm", "tqdm"),
]


def check_and_install():
    missing = []
    for mod_name, pip_name in REQUIRED_PKGS:
        try:
            importlib.import_module(mod_name)
            print(f"  ✅ {mod_name}")
        except ImportError:
            missing.append(pip_name)
            print(f"  ❌ {mod_name}")

    if missing:
        print("\n📦 Eksik paketler kuruluyor...")
        for pkg in missing:
            cmd = [sys.executable, "-m", "pip", "install", pkg]
            if "flash-attn" in pkg:
                cmd += ["--no-build-isolation"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ❌ {pkg} KURULAMADI: {result.stderr[:200]}")
                print(f"  ℹ️  Devam etmek icin: pip install {pkg}")
            else:
                print(f"  ✅ {pkg}")
    else:
        print("\n✅ Tum bagimliliklar hazir")


def main():
    print("=" * 50)
    print("🎤 VoxCPM2 Inference")
    print("=" * 50)

    # 1. Bagimliliklari kontrol et
    print("\n📋 Bagimliliklar kontrol ediliyor...")
    check_and_install()

    # 2. Modeli indir (HF'den)
    print("\n📥 Model indiriliyor (openbmb/VoxCPM2)...")
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(repo_id="openbmb/VoxCPM2")
    print(f"  ✅ {model_path}")

    # 3. Config kontrol
    import json
    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    arch = config.get("architecture", "???")
    print(f"  🏗️  Architecture: {arch}")
    assert arch == "voxcpm2", f"Beklenen: voxcpm2, Alinan: {arch}"

    safetensors = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    print(f"  📦 Safetensors: {len(safetensors)} dosya")

    vae_ok = os.path.exists(os.path.join(model_path, "audiovae.pth"))
    print(f"  🎵 audiovae.pth: {'✅' if vae_ok else '❌'}")
    if not vae_ok:
        print("  ❌ HATA: audiovae.pth bulunamadi!")
        return

    # 4. CUDA kontrol
    import torch
    print(f"\n🔧 CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Inference
    print("\n🎬 Inference basliyor...")
    import asyncio
    import numpy as np
    import time

    from nanovllm_voxcpm import VoxCPM

    async def run_inference():
        server = VoxCPM.from_pretrained(
            model=model_path,
            max_num_batched_tokens=2048,
            max_num_seqs=1,
            max_model_len=2048,
            gpu_memory_utilization=0.90,
            enforce_eager=True,
            devices=[0],
        )
        await server.wait_for_ready()
        print("  ✅ Model hazir!")

        model_info = await server.get_model_info()
        sample_rate = int(model_info["sample_rate"])
        print(f"  🔊 Sample rate: {sample_rate} Hz")

        # Metin
        text = "Merhaba dunya! Bugun hava cok guzel, nasilsiniz?"

        print(f"\n  📝 Metin: {text}")
        print("  🔊 Ses sentezleniyor...")

        buf = []
        start = time.time()
        async for data in server.generate(
            target_text=text,
            cfg_value=2,
            temperature=1.0,
        ):
            buf.append(data)

        wav = np.concatenate(buf, axis=0)
        elapsed = time.time() - start
        dur = wav.shape[0] / sample_rate

        print(f"\n  ⏱️  Sure: {elapsed:.2f}s")
        print(f"  🎵 Ses: {dur:.2f}s")
        print(f"  📊 RTF: {elapsed/dur:.3f}")

        # WAV olarak kaydet
        import soundfile as sf
        out_path = "voxcpm2_output.wav"
        sf.write(out_path, wav, sample_rate)
        print(f"  💾 Kaydedildi: {out_path}")

        await server.stop()
        return wav, sample_rate

    wav, sr = asyncio.run(run_inference())
    print(f"\n✅ TAMAM! Ses dosyasi: voxcpm2_output.wav ({len(wav)} samples @ {sr}Hz)")


if __name__ == "__main__":
    main()
