# IterCAD Training Integration

This directory vendors [ms-swift](https://github.com/modelscope/ms-swift) for
reproducible IterCAD training. The upstream license and documentation are kept
unchanged.

IterCAD adds two training-specific components:

- `examples/train/grpo/plugin/cad_grpo_plugin.py` registers the CAD multi-turn
  scheduler, geometry rewards, and Geometry-Viable Prefix Masking (GVPM).
- `swift/rlhf_trainers/rollout_mixin.py` runs blocking CAD scheduler steps in a
  thread pool so colocated multi-turn rollouts can progress concurrently.

Use the public entry points from the parent directory:

```bash
bash train/IterCAD_SFT.sh
bash train/IterCAD_RL.sh
```

RL training requires the CAD reward server to be started separately in the
CadQuery environment:

```bash
conda activate cq
bash train/IterCAD_Reward_Server.sh
```

Model, dataset, and output locations are configured by the entry scripts and
environment variables. Local checkpoints, generated data, logs, and credentials
must not be committed to this directory.
