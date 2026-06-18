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
- PR q90/q95/q99 intensity bias and neighborhood skill.
- T2M warm- and cold-tail MAE.
- Rank histograms and ensemble-quantile reliability.
- Frequency bias, POD, FAR, and CSI as event diagnostics.
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
development. Extreme-aware sampling and physical tail losses belong in v9.4.

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
- Heavy-rainfall and warm/cold T2M errors improve.
- Reliability and spread/error ratio do not deteriorate.
- Gains are not restricted to one month or one lead.

## v9.4: Extreme-Aware Sampling and Field Losses

v9.4 should start from the best independently validated v9.3 checkpoint and
retain the v9.3 architecture. This version changes the training distribution
and objectives so rare high-impact fields contribute enough gradient without
turning the model into a threshold-specific probability classifier.

### 1. Extreme-Balanced Sampling

Construct the training sampler from labels calculated only from the 1999-2020
training period. Stratify samples by variable and forecast lead using physical
targets:

- Heavy-precipitation samples.
- Warm-extreme T2M samples.
- Cold-extreme T2M samples.
- Ordinary background samples.

Initial batch composition to test:

```text
ordinary background: 65 percent
heavy precipitation: 15 percent
warm T2M extreme:    10 percent
cold T2M extreme:    10 percent
```

Use month-, lead-, and grid-aware training quantiles when defining extremes.
A sample may belong to more than one extreme category, but it must not be
duplicated within a batch.

Implementation requirements:

- Save the sampling labels and thresholds as a versioned training artifact.
- Record the natural and sampled frequency of every category.
- Apply inverse-sampling-probability weights, or an equivalent correction, so
  oversampling does not teach an artificially high extreme-event climatology.
- Keep a configurable fraction of uniformly sampled batches.
- Never use 2021-2024 observations to construct training labels or thresholds.

### 2. Physical Tail-Intensity Loss

Add small auxiliary losses after decoding model output to physical units.
These losses should improve extreme magnitude while retaining the main
flow-matching velocity objective.

For precipitation:

- Smooth-L1 loss with smoothly increasing weight above training q90 and q95.
- Error in the upper spatial quantiles of each forecast field.
- Separate diagnostic losses for q90, q95, and q99 intensity.

For T2M:

- Smooth-L1 loss with smoothly increasing weights below q10/q05 and above
  q90/q95.
- Treat warm and cold tails independently.
- Apply the loss to decoded absolute T2M even when the trained target is a
  GEOS residual.

Use continuous, capped tail weights rather than discontinuous class weights.
Start with a combined physical tail-intensity weight between `0.02` and `0.05`
after the ordinary velocity field has begun to stabilize.

### 3. Neighborhood Extreme Loss

Add a soft neighborhood loss for spatial placement and event extent:

- Apply differentiable soft exceedance thresholds to decoded precipitation.
- Compare observed and forecast neighborhood fractions at 3x3 and 5x5 scales.
- Begin with q90 and q95 precipitation thresholds.
- Use month-, lead-, and grid-specific thresholds from training observations.
- Keep this loss precipitation-only initially; T2M fields are spatially
  smoother and already receive gradient and multi-scale losses.

The neighborhood objective should tolerate a small displacement while still
penalizing missing, excessively broad, or incorrectly located heavy-rainfall
areas. Log each neighborhood scale separately.

### v9.4 Ablation Order

1. Best v9.3 baseline.
2. Extreme-balanced sampling only.
3. Physical tail-intensity loss only.
4. Neighborhood loss only.
5. Balanced sampling plus physical tail-intensity loss.
6. Add the neighborhood loss to the best preceding configuration.

### v9.4 Acceptance Criteria

- Overall PR and T2M CRPS do not deteriorate materially.
- PR q95/q99 intensity bias and neighborhood skill improve.
- Warm- and cold-tail T2M MAE improve.
- Ensemble-mean RMSE and bias remain stable.
- Gains occur across multiple months and leads, not only one event or season.
- Improvements survive year-block or initialization-block bootstrap testing.

## v9.5: GEOS Ensemble and Temporal Evolution Conditioning

v9.5 should start from the best v9.4 configuration and improve the information
supplied by GEOS. The current target-lead ensemble mean discards both forecast
evolution and member disagreement, which are valuable signals for extreme
amplitude and uncertainty.

### 1. GEOS Ensemble Spread and Quantiles

Do not reduce GEOS to only its ensemble mean before model conditioning. For
each PR and T2M lead, provide:

- Ensemble mean.
- Ensemble standard deviation.
- Ensemble q10.
- Ensemble q90.
- Member-count or member-availability mask.

Calculate these statistics before spatial cropping and normalize each statistic
with training-period artifacts. Handle deterministic years explicitly:

- Set ensemble spread to zero.
- Set q10 and q90 equal to the deterministic member.
- Mark the available member count so the model can distinguish a deterministic
  forecast from a genuinely low-spread ensemble.

Keep the raw GEOS members available for verification, but use summary
statistics first. A permutation-invariant member encoder can be evaluated later
only if the summary representation proves insufficient.

### 2. GEOS Temporal Evolution Encoder

Encode the four GEOS leads as an ordered sequence instead of flattening them as
unrelated channels or selecting only the target lead. For the requested target
week, construct:

- Target-lead ensemble statistics.
- Target minus previous-lead tendency.
- Next lead minus target tendency.
- Explicit previous/next availability masks at Week 1 and Week 4 boundaries.
- Lead-position embeddings.

Use a compact shared temporal convolution or temporal-attention encoder for PR
and T2M, followed by variable-specific projections into the local backbone.
Preserve the separate PR/T2M output heads from v9.3.

Implementation requirements:

- Do not use future observations; all temporal inputs must come from the GEOS
  forecast initialized at the same date.
- Preserve physical lead order.
- Log the norm and missing-boundary frequency of every tendency feature.
- Include an ablation that uses ensemble statistics for the target lead only.
- Include an ablation that uses temporal evolution with the ensemble mean only.

### v9.5 Ablation Order

1. Best v9.4 baseline with target-lead GEOS ensemble mean.
2. Target-lead GEOS mean, spread, q10, and q90.
3. GEOS temporal evolution using ensemble mean only.
4. Full ensemble-statistic temporal evolution encoder.

### v9.5 Acceptance Criteria

- Independent PR and T2M CRPS improve or remain stable.
- PR q95/q99 and T2M warm/cold tail errors improve.
- Week 3-4 skill improves without degrading Week 1-2 materially.
- Spread/error ratio and rank histograms improve or remain stable.
- The model uses GEOS spread without simply inflating generated variance.
- Runtime and memory increases are justified by independent forecast skill.

## Reproducibility Requirements

Every version should save:

- Full config in the output directory.
- Git commit and branch.
- Training, validation, and test year ranges.
- Statistics and threshold artifact paths with checksums.
- Noise rho, beta, coarse-kernel, ensemble, and ODE settings.
- Parameter count and peak GPU memory.
- Probabilistic and physical-tail metrics.
- Per-variable, per-tail, per-month, and per-lead metrics.

Do not compare weighted training noise MSE directly across versions when loss
weights, conditioning, or architecture differ. Use a common evaluation script
with fixed samples, flow times, noise seeds, and unweighted PR/T2M per-lead
noise MSE.
