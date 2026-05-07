"""
VoxCPM2 WSL Final Test
"""
import os, sys, json, gc, asyncio, time, numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("VoxCPM2 WSL Final Test")
print("=" * 50)

gc.collect()
import torch

if torch.cuda.is_available():
    torch.cuda.empty_cache()
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from nanovllm_voxcpm import VoxCPM
print("VoxCPM import OK")


async def run_test():
    from huggingface_hub import snapshot_download

    model_path = os.environ.get("MODEL_PATH")
    if not model_path or not os.path.exists(str(model_path)):
        print("\nModel indiriliyor (openbmb/VoxCPM2)...")
        model_path = snapshot_download(repo_id="openbmb/VoxCPM2")
        print(f"Model: {model_path}")

    config = json.load(open(os.path.join(str(model_path), "config.json")))
    arch = config.get("architecture")
    print(f"Architecture: {arch}")
    assert arch == "voxcpm2", f"Beklenen voxcpm2, alinan {arch}"

    print("\nModel yukleniyor...")
    server = VoxCPM.from_pretrained(
        model=str(model_path),
        max_num_batched_tokens=2048,
        max_num_seqs=1,
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        devices=[0],
    )
    await server.wait_for_ready()
    print("Model hazir!")

    model_info = await server.get_model_info()
    sr = int(model_info["sample_rate"])
    print(f"Sample rate: {sr} Hz")

    text = "Merhaba dunya! Bugun hava cok guzel."
    print(f"\nMetin: {text}")

    buf = []
    start = time.time()
    async for data in server.generate(target_text=text, cfg_value=2, temperature=1.0):
        buf.append(data)

    wav = np.concatenate(buf, axis=0)
    elapsed = time.time() - start
    dur = wav.shape[0] / sr

    print(f"\nSure: {elapsed:.2f}s")
    print(f"Ses: {dur:.2f}s")
    print(f"RTF: {elapsed/dur:.3f}")

    import soundfile as sf
    out = "voxcpm2_output_final.wav"
    sf.write(out, wav, sr)
    print(f"Kaydedildi: {out} ({len(wav)} samples @ {sr}Hz)")

    await server.stop()
    print("\nINFERENCE BASARILI!")


if __name__ == "__main__":
    asyncio.run(run_test())
