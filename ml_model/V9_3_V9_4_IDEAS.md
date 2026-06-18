# South Asia Flow v9.3, v9.4, and v9.5 Roadmap

## Reference Model

The reference is v9.2:

- Domain: 55E-100E, 0N-40N at 1-degree resolution.
- PR target and GEOS-residual T2M target.
- Target-lead-only GEOS PR/T2M conditioning.
- Local SM and MJO predictors.
- Global SST, SSS, IVT, Z500 zonal deviation, and U250 predictors.
- EOF-LHS mixed training noise.
- Velocity-only training during epochs 1-25.
- Joint velocity/variance training during epochs 26-54.
- Variance-only training during epochs 55-80, initialized from the best
  pre-variance CRPS checkpoint.
- Lead-loss weights `[1.0, 1.2, 1.5, 2.0]`.

The versions below should be developed sequentially. Each version should start
from the best validated design of the previous version.

## Common Evaluation Protocol

- Training: 1999-2020.
- Model selection: 2021-2023.
- Final untouched test: 2024.
- Physical extreme thresholds: month-, lead-, and grid-specific quantiles from
  raw observed 2001-2020 climatology.
- Apply the same observed threshold to observations, ML members, and GEOS
  members.
- Report PR upper q90/q95, T2M upper q90/q95, and T2M lower q10/q05.
- Treat system-specific self-threshold scores as diagnostic only.

Required metrics:

- CRPS, ensemble-mean RMSE, bias, ACC, and spread/error ratio.
- Brier score and Brier Skill Score.
- Reliability diagrams and Brier decomposition.
- ROC AUC and precision-recall AUC.
- Frequency bias, POD, FAR, and CSI.
- Separate month and lead-week results.

## v9.3: Recommended Architecture Update

v9.3 should implement only the following four changes. Other proposed changes
belong in v9.4 or v9.5.

### 1. Static Terrain, Land, and Coordinate Channels

Add the following static local channels to the South Asia backbone:

- Orography/elevation.
- Land-sea mask.
- Normalized latitude.
- `sin(longitude)`.
- `cos(longitude)`.

Optional terrain slope, aspect, and distance-to-coast channels are not part of
the initial v9.3 implementation.

Rationale:

- Himalayan elevation strongly controls rainfall and temperature.
- The current land/ocean mask only weights the loss; the model does not receive
  it as an input.
- Coordinate channels help the convolutional model distinguish geographic
  locations and regional regimes.

Implementation requirements:

- Crop static inputs to the same 41x46 South Asia target grid.
- Verify exact latitude/longitude alignment with PR, T2M, and GEOS.
- Normalize elevation using training-domain statistics.
- Keep land mask and cyclic coordinate channels bounded.
- Create a new named statistics artifact if elevation statistics are included.
- Plot every static channel as a preflight alignment diagnostic.

### 2. Spatial Global Tokens With Cross-Attention

Replace the current global-context path that compresses all full-global fields
to one 128-value vector through global-average pooling.

Proposed design:

1. Encode global SST, SSS, IVT, Z500 zonal deviation, and U250 with a strided
   convolutional encoder.
2. Preserve a coarse spatial feature grid.
3. Flatten the coarse grid into global tokens.
4. Add latitude/longitude positional encoding to the tokens.
5. Use cross-attention from South Asia UNet bottleneck features to the global
   tokens.
6. Retain a pooled global summary for FiLM conditioning.
7. Use circular padding along longitude in the global encoder.

Rationale:

- Global-average pooling loses the location, sign, and spatial relationship of
  ENSO, IOD, MJO, IVT, and circulation anomalies.
- Cross-attention lets each South Asia feature location select relevant global
  teleconnection information.

Initial implementation constraints:

- Add cross-attention at the UNet bottleneck only.
- Use one or two attention blocks rather than modifying every UNet stage.
- Keep the v9.2 UNet widths `[128, 256, 512, 768]`.
- Log token shape, attention memory, and parameter count.
- Add an ablation flag that restores pooled-vector-only global context.

### 3. Separate PR and T2M Heads

Keep the shared UNet backbone but replace every shared two-channel output head
with independent variable-specific heads:

- PR velocity head.
- T2M velocity head.
- PR variance head.
- T2M variance head.

Use equal head capacity for all four leads in v9.3.

Rationale:

- PR is sparse, nonnegative, intermittent, and spatially sharp.
- T2M is smooth, terrain-sensitive, and strongly related to GEOS bias.
- Shared final decoding can create negative transfer.
- Separate variance heads permit different PR and T2M spread behavior.

Logging requirements:

- PR velocity loss.
- T2M velocity loss.
- PR variance loss.
- T2M variance loss.
- Per-variable gradient and multi-scale losses.

### 4. Small Gradient and Multi-Scale Loss

Retain the existing pixel velocity MSE and add:

- First-order horizontal gradient MSE.
- First-order vertical gradient MSE.
- Coarse-field MSE after 2x average pooling.
- Coarse-field MSE after 4x average pooling.

Suggested initial weights:

```yaml
gradient_loss_weight_pr: 0.03
gradient_loss_weight_t2m: 0.01
multiscale_loss_weight_2x_pr: 0.03
multiscale_loss_weight_2x_t2m: 0.01
multiscale_loss_weight_4x_pr: 0.02
multiscale_loss_weight_4x_t2m: 0.01
```

Rationale:

- Pixel MSE can reward overly smooth rainfall.
- Gradient loss preserves rainfall boundaries.
- Multi-scale loss preserves monsoon-scale and synoptic organization.
- Smaller T2M weights avoid over-sharpening a naturally smoother field.

These losses should operate on the velocity target during initial v9.3
development. Do not add tail-aware Brier loss in the same version.

### What Remains Unchanged in v9.3

- Target-lead-only GEOS conditioning.
- GEOS-residual T2M target.
- Local and global predictor variables.
- EOF-LHS noise generation and rho/beta settings.
- Flow-time schedule.
- Joint velocity/variance training schedule.
- Lead-loss weights.
- Main UNet depth and widths.
- Training, validation, and test year definitions.

### v9.3 Ablation Order

Do not combine all changes without intermediate checks:

1. v9.2 baseline.
2. Static geography only.
3. Separate PR/T2M heads only.
4. Static geography plus separate heads.
5. Add spatial global tokens and bottleneck cross-attention.
6. Add gradient and multi-scale loss.

### v9.3 Acceptance Criteria

Promote v9.3 if:

- 2024 CRPS improves or remains within 2% of v9.2.
- 2024 ensemble-mean RMSE remains within 2% of v9.2.
- PR spatial structure and Week 3-4 skill improve.
- Heavy-rainfall and warm/cold T2M BSS improve without calibration leakage.
- Reliability and spread/error ratio do not deteriorate.
- Gains are not restricted to one month or one lead.

## v9.4: Direct Extreme-Probability Training

v9.4 should start from the best v9.3 architecture. It adds objectives that
directly target extreme-event probability and Brier skill.

### 1. Tail-Aware Differentiable Brier Loss

Add a small proper Brier loss using training-period observed thresholds:

- PR upper q90/q95.
- T2M upper q90/q95.
- T2M lower q10/q05.

Use month-, lead-, and grid-specific thresholds saved as a versioned artifact.

For upper-tail events:

```text
member probability = sigmoid((forecast - threshold) / temperature)
```

For lower-tail events:

```text
member probability = sigmoid((threshold - forecast) / temperature)
```

Average member probabilities and compare with the observed event using squared
probability error.

Suggested schedule:

- Epochs 1-25: velocity training.
- Epochs 26-54: velocity plus variance.
- Epoch 55 onward: variance-only training initialized from the best
  pre-variance CRPS checkpoint.
- Enable tail-aware Brier loss during the joint phase only after the velocity
  forecast is stable; test epoch 35 or 40 as the initial start.

Suggested initial weight:

```yaml
extreme_brier_loss_weight: 0.01
```

Test weights up to `0.03` only after checking CRPS and RMSE.

### 2. Tail-Weighted Variance Training

Give additional variance-loss weight to observed extremes:

```text
normal case: 1.0
q90/q10 event: 1.5
q95/q05 event: 2.0
```

Keep weights configurable separately for:

- PR upper tail.
- T2M warm tail.
- T2M cold tail.

The objective is better ensemble spread on extremes, not a larger deterministic
forecast amplitude.

### 3. Probability Calibration

Fit probability calibration using 2021-2023 only:

- Logistic calibration first.
- Isotonic calibration when sample size is sufficient.
- Separate by variable, tail, month, and lead.

Apply the frozen calibrator to 2024 and report raw and calibrated BSS.

Calibration is required for evaluation but remains post-processing; it must not
use 2024 outcomes during fitting.

### v9.4 Acceptance Criteria

- Positive or materially improved 2024 BSS for PR q95 and T2M q95/q05.
- Reliability curves move toward the diagonal.
- ROC AUC does not decline materially.
- CRPS and RMSE remain within 2% of the best v9.3 model.
- Improvement survives year-block or initialization-block bootstrap testing.

## v9.5: Temporal and Variance Refinements

v9.5 should contain the remaining higher-risk architecture experiments.

### 1. GEOS Temporal Evolution Encoder

Replace target-lead-only GEOS conditioning with a compact temporal
representation containing:

- Target lead.
- Target minus previous lead tendency.
- Next lead minus target tendency.
- Explicit availability masks for Week 1 and Week 4 boundaries.

Use a small temporal convolution or attention encoder. Do not return to simply
flattening all four leads as unrelated channels.

### 2. Lead-Scaled Decoder Capacity

Test larger decoder heads for later leads:

```text
Week 1: 48 hidden channels
Week 2: 48 hidden channels
Week 3: 64 hidden channels
Week 4: 96 hidden channels
```

Apply this separately to PR and T2M heads. Accept the change only if Week 3-4
independent CRPS/BSS improves.

### 3. Bounded Log-Scale Variance Heads

Replace unrestricted positive `softplus` scale with bounded log-scale:

```text
log_scale = clamp(raw_log_scale, minimum, maximum)
scale = exp(log_scale)
```

Use different bounds for PR and T2M. Monitor:

- Fraction saturated at minimum scale.
- Fraction saturated at maximum scale.
- Spread/error ratio.
- Rank histograms.
- Extreme-event reliability.

### v9.5 Acceptance Criteria

- Week 3-4 CRPS, RMSE, and BSS improve.
- Variance bounds do not saturate excessively.
- Rank histograms and spread/error ratio improve.
- Runtime and memory increases are justified by independent forecast skill.

## Reproducibility Requirements

Every version should save:

- Full config in the output directory.
- Git commit and branch.
- Training, validation, calibration, and test year ranges.
- Statistics and threshold artifact paths with checksums.
- Noise rho, beta, coarse-kernel, ensemble, and ODE settings.
- Parameter count and peak GPU memory.
- Raw and calibrated probabilistic metrics.
- Per-variable, per-tail, per-month, and per-lead metrics.

Do not compare weighted training noise MSE directly across versions when loss
weights, conditioning, or architecture differ. Use a common evaluation script
with fixed samples, flow times, noise seeds, and unweighted PR/T2M per-lead
noise MSE.
