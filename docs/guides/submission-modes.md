# Submission modes

A **cell** is one task × encoder × protocol × seed. The task card fixes the raw input (one wrist, six channels in a
forward–left–up frame at 50 Hz, the windows, the splits), the decoder, the recipe, the selection rule and the
scorer. The encoder is the only free part, and there are two ways to supply one.

## Encoder plugin

Any PyTorch module can be evaluated. Write a builder and describe the encoder in a card.

```python
# my_package/imubench.py
import torch, torch.nn as nn

class MyEncoder(nn.Module):
    embed_dim = 768                       # required: an integer attribute

    def forward(self, x):                 # x: [B, C, T] float; acc x/y/z (m/s²) then gyro x/y/z (rad/s); card's model_hz
        ...
        return {"tokens": tokens,         # [B, N, D] over time: read by the pose and odometry decoders
                "pooled": pooled}         # [B, D]: read by the classification and retrieval heads
        # returning only [B, D], or only [B, N, D], is also accepted

def build(weights=None, **kwargs) -> nn.Module:
    model = MyEncoder(**kwargs)
    if weights:
        model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model
```

```yaml
# my_encoder.yaml: one encoder per file, the key is the encoder id
my_encoder:
  name: My encoder (Doe et al., 2026)
  external_encoder: plugin
  entry: my_package.imubench:build          # imported once per job; the clone root is on sys.path
  weights: hf://my-org/my-model/encoder.pt  # or a local path
  channels: 6                               # 3 for accelerometer-only models
  model_hz: 50                              # the harness resamples the 50 Hz input to this rate
  acc_units: m/s2                           # or g
  normalize_method: zscore                  # zscore | physical | raw, applied before your model
  params_m: 85
  pretrain_data: Capture-24 (all), UK Biobank (train)    # the exact corpus, for the overlap audit
  weights_url: https://huggingface.co/my-org/my-model
  track: open
```

What the harness does around your model:

- resamples to `model_hz` (anti-aliased polyphase filter when downsampling, linear interpolation when upsampling);
- converts accelerometer units when `acc_units: g`;
- drops the gyroscope when `channels: 3`;
- applies `normalize_method` to the raw window;
- for models that return a single vector, repeats it over the token grid so that the kinematic decoders still run
  (such a row is scored on every task, with that limitation visible in the result file).

`examples/plugin/inrtlwrist_plugin.{py,yaml}` wraps the reference encoder through this interface and reproduces its
native scores; copy it as a template. The full contract is in `imu_lm/models/baselines/plugin_encoder.py`.

## Built-in architectures

An encoder trained with this repository's trainer (the shared ViT1D of the controlled track) is registered with a run
directory or checkpoint instead of a plugin entry; the cards of the released encoders in
`bench/registry/encoders.yaml` show the fields. This is the path for controlled-track submissions, where the backbone
and the corpus are fixed and only the pretraining objective differs. See [Reproducing the paper](reproducing.md) for
the pretraining command.

## Embeddings instead of an encoder

Not supported, by design:

- the fine-tune protocol needs the encoder's weights;
- the pose and odometry decoders need per-patch tokens, not one vector per window;
- a frozen-probe-only entry would be ranked against rows that ran both protocols.

If the weights cannot be made public, release them to the maintainers privately with the submission. A
frozen-probe-only embeddings mode for the global tasks, listed outside the ranking, is on the [roadmap](../roadmap.md).
