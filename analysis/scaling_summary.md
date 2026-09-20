# Does pretraining-corpus scale matter? Four axes, verified 2026-09-10

Every number below is read straight out of `bench/results/` by
`analysis/volume_diversity/` and re-derived on 2026-09-10. Single seed (42)
throughout — see **Reliability** for what that costs.

---

## 1. What was varied

Four data axes, each isolated so that only one thing moves. All arms share the
same encoder (ViT1D alt-attn, 384-d, 12 layers), the same z-score input, the
same 50,000-step budget, and the same frozen-probe protocol on the same four
tasks — so a smaller arm simply revisits its windows more often. Matched
compute, not matched epochs, is the control the claim needs.

| Axis | Range swept | Arms | Status |
|---|---|---|---|
| **Volume** | 12,500 → 1,738,094 windows (18 h → 2,472 h, **137×**) | `vd_{nar,one,nat,bal}_n{12k,50k,200k,800k}` + 2 anchors | ✅ 18/18 |
| **Source count** | 1 → 4 datasets (effective types 158 → 1000) | `vd_src{1,2,3,4}_n50k` | ✅ 4/4 |
| **Motion types** | 3 → 673 effective types (**224×**) | `vd_typ{3,30,300,all}_n12k` | 🔄 3.25/4 |
| **Wearers** | 1 / 3 / 10 / 22 subjects, **OCDetect only** | `vd_sub{1,3,10,all}_n50k` | 🔄 arms built, 4 pretrains + 16 probes submitted 2026-09-10 |

The source ladder deliberately adds datasets **smallest-first**
(motionx → sonic → eatingspeed → ocdetect), so the 1-source arm is *not* simply
"the big dataset again". That choice works against the hypothesis that more
sources help.

The type ladder runs at N=12,500 rather than 50,000 because ocdetect's three
largest motion-type clusters hold only 15,065 windows between them — `typ3` is
infeasible at 50k. 12,500 is the factorial's own smallest volume level, so the
type arms anchor against `vd_nar_n12k` / `vd_one_n12k`.

The wearer ladder is confined to **OCDetect (22 subjects)**, because it is the
only source in the pool that exposes a wearer identity at all: motionx and sonic
expose one *synthetic clip* per "session" and have no subject concept, and
eatingspeed exposes no provenance through the cached path. Confining the ladder
to one source removes the source confound — only the number of people producing
a fixed 50,000 windows changes — but it also bounds what the axis can support:
free-living wrist data from at most 22 wearers. A wearer effect found here
**cannot** be generalised to "wearer diversity transfers" across corpora.

The realised arms, with the diversity that was actually achieved:

| arm | wearers | effective wearers | effective phys. types |
|---|---:|---:|---:|
| `vd_sub1_n50k` | 1 | 1.00 | 487.4 |
| `vd_sub3_n50k` | 3 | 2.98 | 570.4 |
| `vd_sub10_n50k` | 10 | 9.58 | 606.5 |
| `vd_suball_n50k` | 22 | 19.53 | 620.4 |

Type-matching held the type axis *nearly* but not exactly fixed: effective types
still drift 487 → 620 (1.27×) as wearers go 1 → 22, because fewer wearers really
do produce fewer motion types and the equal-per-type water-fill can only
redistribute what a subject actually recorded. That residual confound is
**bounded by an already-measured null**: the type axis was swept over 3 → 673
effective types (224×) and moved nothing, so a 1.27× drift cannot plausibly
carry a wearer effect. State it that way rather than claiming clean isolation.

---

## 2. Result: every data axis is flat

Spread = max − min across the arms of one axis, frozen probe, seed 42.

| Axis | arm pose (cm ↓) | odometry ATE (↓) | HAR-WISDM F1 (↑) | microgesture F1 (↑) |
|---|---|---|---|---|
| **Volume** (137×) | 0.130 | 0.0025 | 0.054 | 0.078 |
| **Source count** (1→4) | 0.085 | 0.0067 | 0.035 | 0.136 |
| **Motion types** (224×) | 0.073 | 0.0032 | 0.037 | 0.034 |

Not one axis is monotone, and each axis's optimum lands at a different,
essentially arbitrary position:

| Axis | where the best arm sits |
|---|---|
| Volume | **18 h** is best on arm pose, WISDM and microgesture |
| Source count | **1 source** best on arm pose, **2** on WISDM, **3** on odometry and microgesture |
| Motion types | **300 types** best on WISDM and microgesture, **3 types** best on odometry |

Individual points that make the shape concrete:

- Volume, `nat` column, microgesture: 0.568 (18 h) → 0.522 → 0.549 → 0.531 →
  **0.490** (2,472 h). Monotone *downward* — 137× the data, 8 F1 points worse.
- Source count, microgesture: 0.454 (1 src) → 0.503 → **0.590** (3 src) →
  0.474 (4 src). The full 4-source pool is worse than 3.
- Motion types, arm pose: `typ3` (3 types) scores 8.109, `typall` (673 types)
  scores 8.168. 224× the motion vocabulary, no gain.

---

## 3. Comparison against the recipe factors — with the caveat that matters

Same encoder family, same corpus, same protocol; only the recipe changes.

**HAR mean over 7 datasets** (the noise-averaged statistic, most reliable):

| Factor | spread | best → worst |
|---|---|---|
| **Pretraining objective** | **0.157** | MAE 0.656 › RelCon 0.627 › SimMTM 0.593 › FOCAL 0.499 |
| **Input scaling** | **0.121** | z-score 0.656 › physical 0.589 › phys+patchnorm 0.582 › raw/DC 0.535 |
| Modality (acc+gyro vs acc-only) | 0.024 | 0.656 vs 0.632 |

⚠️ **This is not a like-for-like comparison.** The recipe factors are measured
on a **7-dataset mean**, which averages noise down; the data axes are measured
on **single tasks**, which does not. Comparing 0.157 against 0.054 overstates
the gap. The voldiv arms only ran a 4-task panel, so a 7-dataset mean cannot be
computed for them.

**The clean apples-to-apples comparison is odometry**, where both sides are a
single task:

| | odometry ATE spread |
|---|---|
| Volume axis (137×) | 0.0025 |
| Type axis (224×) | 0.0032 |
| Source axis (1→4) | 0.0067 |
| **Modality** | **0.0261** |
| **Input scaling** | **0.0506** |
| **Objective** | **0.0523** |

Here recipe beats data by **4–20×** with no statistical sleight of hand. This is
the strongest single piece of evidence in the whole study.

**Where the separation does *not* hold:**

- **microgesture** — the source axis (0.136) is the same size as input scaling
  (0.142) and larger than modality (0.092). Data and recipe are not separated
  on this task.
- **arm pose** — the measured run-to-run noise band is **0.108 cm** (three runs
  of a byte-identical config: 8.054 / 8.136 / 8.162). Every data axis
  (0.073–0.130) sits at or below that. So do the objective (0.127) and input
  scaling (0.142) factors. Only modality (0.325) clearly clears it. On this task
  the honest statement is that **nothing is resolved**, recipe included.

---

## 4. What can and cannot be claimed

**Supported:**

> Over the range this corpus spans — 18 h to 2,472 h, 1 to 4 sources, 3 to 673
> motion types — pretraining-corpus scale does not drive downstream performance.
> On odometry, where the comparison is single-task on both sides, the
> pretraining objective and the input representation each move the metric 4–20×
> further than any data axis.

**Not supported (and previously overstated):**

- ❌ "Recipe beats data by an order of magnitude." True on odometry; false on
  microgesture; unresolvable on arm pose. The 10× figure came from comparing a
  7-dataset mean against a single dataset.
- ❌ "Less data is better." The 18 h arm winning three of four tasks is inside
  the noise band on arm pose and is a single seed everywhere. It says the curve
  is flat, not that it slopes down.
- ❌ Anything about corpora larger than 2,472 h, or about acc-only corpora.
  NHANES (4,000 h) and Capture24 (3,122 h) are on disk and untested here.

**Two independent corroborations, both pre-dating this factorial:**

1. A causal contrast at fixed N showed a deliberately-redundant subset collapse
   (WISDM 61.7 → 35.3) while random ≈ diverse (61.7 vs 61.5). So diversity
   matters only at the pathological end — and any real corpus sampled at random
   is already past it.
2. A 5-point volume sweep (1.3% → 100% of an 11-source pool, 300k steps) had the
   **smallest** fraction winning 11 of 16 downstream cells, while the pretraining
   reconstruction loss improved monotonically. Reconstruction quality and
   transfer quality are decoupled.

---

## 5. Reliability

- **Single seed (42)** on all 791 bench cells. Differences below the noise band
  are not interpretable.
- **Measured noise**, arm pose, identical config: 8.054 / 8.136 / 8.162 → spread
  **0.108 cm**. Two of those three runs were later found to have had a second
  SLURM job writing into the same run directory; the clean pair spans 0.026 cm.
  Treat 0.03–0.11 cm as the plausible band.
- **Spread is a crude statistic** — it uses only two points and grows with the
  number of arms. Both sides here use 4–5 arms, so the comparison is fair, but
  these are not formal effect-size estimates. A proper two-way variance
  decomposition needs replicate seeds.
- **Motion types are k-means clusters** in an encoder-free 40-dimensional
  rotation-invariant physical feature space (K=1000), *not* human labels — the
  four sources have non-alignable label vocabularies. An encoder-space
  clustering is reported alongside as an independent read-out.
- **The wearer axis is missing**, and it is the one axis that maps directly onto
  how every downstream task is split (subject-disjoint). Until it lands, the
  claim covers what/how-much data, not who produced it. When it does land it
  will be OCDetect-only (see §1) — a clean within-source contrast, but capped at
  22 wearers of free-living data.

---

## 6. Reproducing

```bash
python analysis/volume_diversity/06_harvest.py
python analysis/volume_diversity/07_analyze.py
```

Arms and their measured diversity: `analysis/volume_diversity/out/arms.json`.
Per-cell results: `bench/results/<task>/vd_*__probe__s42.json`.
