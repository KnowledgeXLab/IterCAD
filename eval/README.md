# IterCAD Evaluation

`evalution.py` provides the agentic multi-turn evaluation pipeline for **IterCAD-Draw** and **IterCAD-Edit**. Each turn executes the generated CadQuery code and returns execution or rendering feedback to the model.

## Prerequisites

- Python 3.8+
- CadQuery, OpenAI SDK, PIL, numpy, trimesh, scipy, matplotlib
- An OpenAI-compatible model API

## Benchmarks

| Task | Default JSONL | Samples |
|------|---------------|---------|
| IterCAD-Draw | `IterCAD_data/benchmark/IterCAD-Draw_1k/IterCAD-Draw_1k.jsonl` | 1000 |
| IterCAD-Edit | `IterCAD_data/benchmark/IterCAD-Edit_200/IterCAD-Edit_200.jsonl` | 200 |

- **Draw**: image → CadQuery code
- **Edit**: text instruction + input code → modified code (no visual edit)

Paths such as `images/...`, `code/...`, and `stl/...` inside a JSONL file are resolved relative to that JSONL file's directory. This also applies to custom files supplied with `--test_samples_IterCAD-Draw` or `--test_samples_IterCAD-Edit`.

## API Configuration

The evaluator reads these environment variables:

- `GENERATOR_API`: OpenAI-compatible base URL; defaults to `http://127.0.0.1:8000/v1`
- `OPENAI_API_KEY`: API key; defaults to `EMPTY`
- `GEN_MODEL`: generator model name; required unless `--gen_model` is supplied

The CLI options `--generator_api`, `--generator_api_key`, and `--gen_model` override their corresponding environment variables when explicitly supplied.

## Quick Start

From the repository root:

```bash
export GENERATOR_API=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
export GEN_MODEL=YOUR_MODEL_NAME

# IterCAD-Draw
python eval/evalution.py \
  --task_type IterCAD-Draw \
  --run_id Draw_1k

# IterCAD-Edit
python eval/evalution.py \
  --task_type IterCAD-Edit \
  --run_id Edit_200
```

Alternatively, from the `eval/` directory:

```bash
python evalution.py --task_type IterCAD-Draw --gen_model YOUR_MODEL
```

To use custom benchmarks:

```bash
python eval/evalution.py \
  --task_type both \
  --test_samples_IterCAD-Draw /path/to/draw/benchmark.jsonl \
  --test_samples_IterCAD-Edit /path/to/edit/benchmark.jsonl \
  --generator_api http://127.0.0.1:8000/v1 \
  --generator_api_key EMPTY \
  --gen_model YOUR_MODEL
```

Optional vLLM parameter:

```bash
--extra_body '{"chat_template_kwargs":{"enable_thinking":false}}'
```

## Key Arguments

| Argument | Default | Notes |
|----------|---------|-------|
| `--task_type` | `IterCAD-Draw` | Also accepts `IterCAD-Edit` and `both` |
| `--run_id` | timestamp | Use a fixed value for resume |
| `--max_workers` | 4 | Keep ≤8 to reduce CadQuery/OCC crashes |
| `--max_turns` | 5 | Maximum number of agentic feedback turns |
| `--pass_k` | 2 | Run multiple attempts and retain the best |
| `--output_dir` | `results_{model}_unified` | Output directory |

## Outputs and Metrics

Under `--output_dir`:

```text
results_{model}_{task}_{run_id}.jsonl
failed_{model}_{task}_{run_id}.jsonl
metrics_{model}_{task}_{run_id}.json
```

- **Successful**: last turn is `exec_success` (with CD) or `done_by_model`
- **mean_cd / median_cd**: computed over successful samples
- **AUC-TR**: normalized area under the CD-tolerance recall curve (CD ∈ [1e-5, 1e-1])
- **avg_turns**: mean model reply count over all samples

## Resume and Pool Recovery

Use the same `--run_id`, `--gen_model`, and `--output_dir` to resume. Completed `uid`s are skipped automatically. The process pool restarts automatically after a CadQuery/OCC crash unless `--no_pool_recover` is set.

## File Layout

```text
eval/
├── evalution.py
├── code_executor.py
├── gen_view.py
└── README.md
```
