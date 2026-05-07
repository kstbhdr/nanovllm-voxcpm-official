# VoxCPM2 Colab & WSL Kullanım Kılavuzu

## Colab (Önerilen)
**Tek tıkla:**
https://colab.research.google.com/github/kstbhdr/nanovllm-voxcpm-official/blob/main/colab_voxcpm2_inference.ipynb

**Runtime → Change runtime type → T4 GPU seç → ▶️ Çalıştır**

## WSL (Ubuntu + NVIDIA GPU)
```bash
cd ~/nanovllm-voxcpm-official
python3.10 -m venv venv && source venv/bin/activate
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
pip install -r requirements.txt && pip install -e .
python voxcpm2_run.py
```

## Docker
```bash
./docker_build_and_run.sh           # build + run
./docker_build_and_run.sh --push    # build + Docker Hub push
```

## Bağımlılıklar
- torch 2.5.1, Python 3.10-3.12
- flash-attn, transformers>=4.51.0, triton>=3.0.0
- soundfile, librosa, numpy, tqdm, xxhash, pydantic, torchcodec, psutil

## Önemli Notlar
1. VRAM fix fork'a özeldir (upstream a710128'de hatalı)
2. Scheduler fix ve deadlock fix fork'a dahildir
3. İlk çalıştırmada flash-attn derlenir (~3-4 dk)
4. openbmb/VoxCPM2 modeli ~2.5 GB indirilir
