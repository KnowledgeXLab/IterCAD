# Installation

IterCAD uses two Conda environments. Run the following commands from the
repository root.

## Training environment

Used for SFT and RL training.

```bash
conda create -n itercad python=3.12 -y
conda activate itercad

pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install einops \
  "deepspeed>=0.16,<0.19" \
  "accelerate>=1.0" \
  "peft>=0.11,<0.19" \
  "datasets>=3.0,<4.0"

pip install -e train/ms-swift
pip install "qwen_vl_utils>=0.0.14" "transformers==5.2.0"
pip install vllm==0.17.0

pip install --no-deps \
  "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```

Before training, adjust the model path, GPU count, and batch size in the
training scripts as needed.

## Evaluation and reward-server environment

Used for benchmark evaluation and the CAD reward server.

```bash
conda create -n cadquery -y --override-channels -c conda-forge \
  python=3.12 cadquery numpy scipy trimesh matplotlib pillow

conda activate cadquery
pip install openai fastapi uvicorn
```

Use `itercad` for training and `cadquery` for evaluation or reward serving.
