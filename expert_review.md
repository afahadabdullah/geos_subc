# Expert Review Follow-Up: Diffusion/Flow Lessons for This Project

## Context From Expert Feedback

The expert highlighted two generic lessons from adapting modern image-domain diffusion models to climate fields:

1. **The noise schedule must be retuned for climate data.** RGB image defaults do not transfer cleanly to geophysical variables because precipitation, radiation, temperature, winds, and ocean/state variables have very different spectra and dynamic ranges. A model must be trained so that the forward/noising process fully destroys the recoverable signal before learning the reverse process.

2. **Overfitting to large-scale climate modes is a real failure mode.** In their example, the model showed too little ensemble dispersion during the training period and suspicious phase coherence with observed Arctic Oscillation behavior. Their mitigation was to regularize the earliest denoising stages, where large-scale red-spectrum modes emerge first, including use of different checkpoints at different denoising stages.

Those lessons are directly relevant to this work, but our implementation is not a literal EDM/DDPM pipeline in the active path. Our active multiv1 workflow is a rectified-flow / flow-matching model, so the same ideas translate into:

- the choice of starting noise prior `x0`,
- the distribution over flow time `t`,
- per-channel target transforms and scaling,
- whether training, validation, and generation use the same noise manifold,
- and how we evaluate memorization of large-scale climate modes.

## Current Implementation Mapping

The active multi-target model is `FlowMatchingModel` plus `CustomFlowMatcher` in `ml_model/flow_matching_multi.py`.

Important current behavior:

- Training samples time as `t ~ Uniform(0, 1)` in `CustomFlowMatcher.sample_time_batch`.
- Main velocity training starts from pure Gaussian noise via `torch.randn_like(target_norm)` in `ml_model/train_flow_multiv1.py`.
- Validation and generation often use EOF-LHS structured noise plus variance scaling, controlled by `validation_rho_pr`, `validation_rho_t2m`, `validation_var_beta_pr`, and `validation_var_beta_t2m` in `ml_model/config_flow_multiv1.yaml`.
- Inference/generation toggles between pure Gaussian and EOF-LHS-plus-variance in `ml_model/generate_multiv1_ensembles.py`.
- Comparison scripts already sweep pure Gaussian checkpoints and EOF/rho/beta strategies, especially `ml_model/compare_noise_multi.py` and `ml_model/compare_noise_v4_ckpts_multi.py`.

So the expert's "retune the noise schedule" advice applies, but for us it should not be implemented by blindly copying an EDM sigma schedule. It should be implemented by calibrating the flow path and endpoint prior.

## Main Issue For Our Work

The largest technical mismatch is:

> We train the velocity field mostly from pure Gaussian `x0`, but often evaluate and generate from EOF-structured `x0`.

That means the model is asked at inference time to integrate trajectories from a starting manifold it did not consistently see during training. The fact that EOF-LHS sometimes improves CRPS is useful evidence that physically structured perturbations are helpful, but it also means our current training objective and inference prior are not cleanly aligned.

This is the flow-matching analogue of using an uncalibrated diffusion noise schedule.

## What Needs To Change

### 1. Fix The Evaluation Split First

Current config trains through 2020:

```yaml
train_start_year: 1999
train_end_year: 2020
val_start_year: 2021
val_end_year: 2023
```

But several generation/evaluation wrappers refer to "held-out 2020-2021" or generate 2020-2021 pure-noise forecasts. If 2020 is included in training, it should not be described as held out.

We should choose one clean protocol:

- **Option A:** train on 1999-2019, validate/test on 2020-2021.
- **Option B:** train on 1999-2020, validate/test on 2021-2023.

For expert-facing results, Option A is cleaner because many current scripts already frame 2020-2021 as the evaluation period. If we keep Option B, then all "held-out 2020-2021" labels and scripts need to be renamed or adjusted.

This also affects global stats and climatology files. If we use 2020 as held out, stats/climatologies should be rebuilt using only the training period where appropriate.

### 2. Make The Noise Policy A First-Class Config

Right now, training, validation, comparison, and generation each partly define their own noise behavior. We should centralize this.

Recommended config additions:

```yaml
train_noise_mode: gaussian          # gaussian | eof_lhs_mix | scheduled_mix
train_rho_pr: 0.0
train_rho_t2m: 0.0
train_eof_probability: 0.0
train_time_schedule: uniform        # uniform | beta | endpoint_weighted | stratified
train_time_beta_alpha: 1.0
train_time_beta_beta: 1.0
```

Then validation/generation should declare whether they intentionally match or intentionally differ from training:

```yaml
validation_noise_mode: eof_lhs_mix
validation_rho_pr: 0.15
validation_rho_t2m: 0.05
```

This makes the experimental contract explicit.

### 3. Start With A Conservative Noise-Prior Sweep

We should not jump directly to a 98 percent EOF / 2 percent Gaussian prior. Our current best-looking validation settings are much more tempered:

```yaml
validation_rho_pr: 0.15
validation_rho_t2m: 0.05
validation_var_beta_pr: 0.3
validation_var_beta_t2m: 0.01
```

That suggests the useful EOF signal is present, but strong EOF structure can easily reduce dispersion or over-impose teleconnection phase.

Recommended sweep:

- pure Gaussian training: `rho_pr=0.0`, `rho_t2m=0.0`
- light EOF mix: `rho_pr=0.05`, `rho_t2m=0.02`
- current validation-like mix: `rho_pr=0.15`, `rho_t2m=0.05`
- stronger PR-only or asymmetric mix: PR higher than T2M

The goal is not just lower CRPS; it is lower CRPS without phase locking and without collapsed dispersion.

### 4. Retune The Flow-Time Schedule

In diffusion language, this is the "noise schedule" problem. In our rectified-flow code, the direct analogue is the sampling/weighting of `t` during training.

Current behavior is uniform over `[0, 1]`. We should test alternatives that put more training pressure near the early reverse trajectory, where large-scale modes emerge:

- stratified `t` bins to guarantee coverage of early/middle/late flow,
- beta-distributed `t` with more mass near `t=0`,
- endpoint-weighted loss, especially near early denoising/flow steps,
- per-channel loss weights so PR and T2M do not compete on incompatible physical scales.

This is the closest equivalent to the expert's "careful log-uniform schedule" advice.

### 5. Add Large-Scale Mode Overfit Diagnostics

The expert's key overfitting diagnostic was not just validation loss. It was whether the generated ensemble had unrealistic phase coherence with observed large-scale climate modes.

For us, we should add diagnostics for:

- NAO/AO projection of generated fields versus observed phase,
- MJO phase/amplitude consistency without future leakage,
- ENSO-conditioned spread,
- training-period dispersion versus validation-period dispersion,
- ensemble spread-skill ratio by lead and region,
- temporal phase coherence of generated anomalies against observations.

A warning sign would be:

- lower training-period spread than validation-period spread,
- generated ensemble mean tracking observed NAO/AO/MJO phase too closely,
- EOF-initialized generations showing too little independent ensemble variability,
- improvements concentrated only in training years or climatologically easy regimes.

This is especially important because our conditioning includes GEOS fields, SST/SSS/soil moisture, MJO wave fields, and teleconnection-dependent EOF perturbations. Those are scientifically meaningful predictors, but they also create possible shortcuts if not audited carefully.

### 6. Consider Checkpoint Mixing Across Flow Steps

The expert's trick of using different checkpoints at different denoising stages maps naturally to our Euler integration.

Instead of one checkpoint for all `num_steps`, we can evaluate:

- early flow steps from an earlier or more regularized checkpoint,
- later flow steps from the best CRPS checkpoint,
- pure Gaussian checkpoint for early steps plus EOF-tuned checkpoint for later steps,
- variance-head disabled or heavily damped during early steps.

This is already close to our current checkpoint-sweep machinery in `compare_noise_v4_ckpts_multi.py`, but we need a sampler that can switch checkpoint/model by step.

Potential schedule:

```text
steps 0-3: earlier checkpoint, stronger regularization, no/low variance scaling
steps 4-9: best validation checkpoint
```

This directly targets the large-scale-mode overfitting issue because early flow steps control broad planetary-scale structure.

### 7. Revisit T2M Target Definition

T2M is currently scaled as absolute temperature over a broad physical range. That can let seasonal/global structure dominate the learning problem.

We should consider:

- T2M anomaly relative to weekly climatology,
- T2M residual relative to GEOS,
- lead/month-specific normalization,
- separate loss balancing for PR and T2M.

The dataset already has a `t2m_target_mode` hook for `absolute` versus `geos_residual`, but the end-to-end decode/evaluation path must be kept consistent before using it in production runs.

## Recommended Near-Term Plan

1. **Clean the split.** Decide whether 2020 is training or held out, then update config, generation scripts, evaluation labels, stats, and climatologies accordingly.

2. **Centralize noise policy.** Add one shared helper/config path for Gaussian versus EOF-LHS mixed noise so training, validation, comparison, and generation can intentionally match.

3. **Run a small training-prior sweep.** Compare Gaussian training against low-rho EOF training while keeping validation fixed.

4. **Add phase-coherence diagnostics.** Evaluate not only CRPS/RMSE, but generated NAO/AO/MJO/ENSO projections and train-vs-validation dispersion.

5. **Test checkpoint mixing.** Add sampler support for early-step and late-step checkpoint selection.

6. **Only then consider aggressive EOF training.** A 98/2 EOF/Gaussian blend may be useful, but it should be treated as an upper-end ablation, not the default starting point.

## Bottom Line

The expert's feedback strongly supports what our experiments are already hinting at: the model is sensitive to the noise/prior structure, and climate-mode overfitting is a central risk.

For this project, the right response is not simply "use EDM." The right response is:

- align the flow training prior with the inference prior,
- calibrate the effective noise/time schedule per variable,
- enforce clean held-out splits,
- and add climate-mode overfit diagnostics that can catch suspicious phase coherence even when scalar validation metrics look good.

That gives us a scientifically defensible path to improve the model without accidentally improving scores by leaking or memorizing large-scale modes.
