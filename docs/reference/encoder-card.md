# Encoder cards

An encoder card is a YAML mapping with one key, the encoder id. Cards of the released encoders are in
`bench/registry/encoders.yaml`; cards registered with `imu-bench eval --encoder <file>` go to
`bench/registry/encoders.d/<id>.yaml`.

## Fields

| Field | Required | Meaning |
|---|---|---|
| `name` | yes | display name, with the reference |
| `params_m` | yes | parameter count in millions |
| `pretrain_data` | yes | the pretraining corpus, dataset by dataset with the split used (audited against the evaluation sets) |
| `normalize_method` | yes | `zscore` (per channel and window), `physical` (m/s² and rad/s as stored) or `raw`; applied by the harness before the encoder, identically in every cell |
| `channels` | yes | `6` (accelerometer + gyroscope) or `3` (accelerometer only) |
| `track` | yes | `open` or `controlled` |
| `external_encoder` | plugin / adapters | `plugin` for the plugin interface; `harnet`, `normwear`, `inertia1` … for the built-in adapters |
| `entry` | plugin | `package.module:build`; `build(weights=<path or None>, **kwargs)` returns an `nn.Module` with an integer `embed_dim` |
| `weights` | plugin | local path, `hf://<org>/<repo>/<file>` (downloaded with `huggingface_hub`), or `builtin` |
| `kwargs` | plugin | keyword arguments passed to `build` |
| `model_hz` | plugin | sampling rate the model expects; the harness resamples from 50 Hz |
| `acc_units` | plugin | `m/s2` (default) or `g` |
| `weights_url` | open track | where the weights are published |
| `run`, `ckpt` | built-in | run directory under `runs/` and checkpoint name (`best`, `latest`, …) for encoders trained with this repository's trainer |
| `alt_attention`, `shared_kernel`, `use_cls` | built-in | architecture flags of the shared ViT1D |
| `preprocess_from` | built-in | pretraining config whose preprocessing the cells inherit |
| `group`, `order`, `short`, `hide_in_summary` | optional | display metadata for the tables |

## Plugin contract

```python
def build(weights=None, **kwargs) -> torch.nn.Module   # attribute embed_dim: int
module(x)   # x: [B, C, T] float at model_hz; acc x/y/z then gyro x/y/z, wrist FLU frame
            # -> {"tokens": [B, N, D], "pooled": [B, D]}, or only [B, N, D], or only [B, D]
```

`tokens` are read by the pose and odometry decoders on the harness grid of `T // 16` patches (0.32 s at 50 Hz);
`pooled` is read by the classification and retrieval heads. A model that returns only a pooled vector is repeated over
the token grid; a model that returns only tokens is mean-pooled. See `examples/plugin/inrtlwrist_plugin.py`.
