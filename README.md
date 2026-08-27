# IterCAD

IterCAD provides supervised fine-tuning (SFT), reinforcement learning,
and agentic evaluation for CadQuery code generation and editing.

- Dataset: [KnowledgeXLab/IterCAD_Data](https://huggingface.co/datasets/KnowledgeXLab/IterCAD_Data)
- Model: [KnowledgeXLab/IterCAD-RL](https://huggingface.co/KnowledgeXLab/IterCAD-RL)

## Setup

Use two isolated Conda environments:

- `itercad`: SFT and RL training
- `cadquery`: evaluation and the CAD reward server

See [docs/INSTALL.md](docs/INSTALL.md) for installation and verification.

Run all commands below from the repository root.

## Training

SFT (activate the training environment first):

```bash
conda activate itercad
# Set MODEL_PATH and other machine-specific options in the script first.
bash train/IterCAD_SFT.sh
```

RL uses the training environment plus a reward server running in the CadQuery
environment. Start the server in a second terminal:

```bash
conda activate cadquery
bash train/IterCAD_Reward_Server.sh
```

Then start RL training:

```bash
conda activate itercad
export CAD_REWARD_API_URL=http://127.0.0.1:8765
# MODEL_PATH, NPROC_PER_NODE, and OUTPUT_DIR may be overridden as needed.
bash train/IterCAD_RL.sh
```

## Evaluation

Start an OpenAI-compatible model server, then run:

```bash
conda activate cadquery
export GENERATOR_API=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
export GEN_MODEL=YOUR_MODEL_NAME

python eval/evalution.py --task_type IterCAD-Draw --run_id draw_eval
python eval/evalution.py --task_type IterCAD-Edit --run_id edit_eval
```

Evaluation results are written under `results_<model>_unified/` by default.
See [eval/README.md](eval/README.md) for benchmark paths, metrics, resume, and
advanced arguments.

## License

IterCAD is released under the [Apache License 2.0](LICENSE). The bundled
`ms-swift` source retains its own license and attribution notices.
