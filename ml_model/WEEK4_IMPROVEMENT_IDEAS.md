# Strategies to Improve Week 4 Prediction Without Degrading Earlier Weeks

> This document catalogs architectural, data, and training strategies for pushing subseasonal (Week 3–4) precipitation forecast skill. Each strategy is designed to avoid the "Capacity Trade-off" problem where improving long-lead accuracy cannibalizes short-lead filters.

## The Core Problem

The UNet has a finite number of convolutional filters. When forced (via loss weighting) to care disproportionately about Week 4, it actively deletes filters tracking fast-moving Week 1 storm fronts and replaces them with slow-moving planetary wave filters (MJO/Z500). We need strategies that give Week 4 its own capacity or better information, without stealing from Weeks 1–2.

---

## Strategy 1: Dedicated Output Heads (Multi-Task Learning)

**Status: ✅ IMPLEMENTED (ml_output_flow3)**

The UNet outputs 64 intermediate features. Four separate `Conv2d(64, 1, 1)` heads each specialize in one forecast week. The shared encoder learns global atmospheric representations; each head learns timescale-specific precipitation patterns. Zero gradient competition between weeks.

**Literature:** DUNE (Deep UNet++-based Ensemble), FuXi-S2S implicit multi-scale decoding.

---

## Strategy 2: FiLM / Cross-Attention Gating

Instead of broadcasting `lead_val` as a flat spatial channel, inject lead-time information via **Feature-wise Linear Modulation (FiLM)**. A small MLP takes the lead index and produces per-channel scale+shift parameters applied after each GroupNorm layer inside the UNet.

This lets the network learn genuinely different feature extraction strategies for Week 1 vs Week 4 at every layer of the network, not just at the output.

For MJO specifically: implement a **Cross-Attention block** in the bottleneck where the query is the UNet features and the key/value come from MJO + Z500. The attention mechanism learns to "focus on MJO when predicting Week 4" and "ignore MJO when predicting Week 1."

**Literature:** ModAFNO (Modulated Adaptive Fourier Neural Operator, 2024), FiLM conditioning.

---

## Strategy 3: Curriculum Learning (Gradual Difficulty Ramp)

Train the model in phases:
- **Epochs 0–200**: Equal weights `[1.0, 1.0, 1.0, 1.0]` — build perfect spatial fundamentals.
- **Epochs 200–500**: Gradually ramp to `[1.0, 1.1, 1.2, 1.3]` — gentle Week 4 nudge.
- **Epochs 500+**: Optionally push to `[1.0, 1.1, 1.3, 1.5]` once early weeks are locked in.

Alternatively, train exclusively on Weeks 1–2 first, freeze early layers, then introduce Weeks 3–4.

**Soft variant:** `w4 = 1.0 + 0.3 * min(epoch / 200, 1.0)` — continuous ramp.

**Literature:** FuXi-S2S (progressive horizon training), standard deep learning curriculum.

---

## Strategy 4: MJO / Rossby Wave Bottleneck Shortcuts

Currently, MJO (2 scalars) is broadcast across `[181, 360]` and passes through 4 downsampling layers, getting diluted by spatial convolutions. Instead, inject MJO and Z500 phase directly into the **UNet bottleneck** as a dense conditioning vector via AdaGN (Adaptive Group Normalization).

This ensures subseasonal planetary drivers are the strongest signal at the network's core — exactly what Week 4 needs.

**Literature:** FuXi-S2S (MJO teleconnection preservation through deep layers).

---

## Strategy 5: FuXi-Style Flow-Dependent Perturbations

Train a small auxiliary network to predict the optimal noise covariance structure from the conditioning variables, rather than using isotropic Gaussian noise or the basic GEOS-variance scaling currently in `test_flow.py`. The ensemble spread becomes physically informed at every pixel and lead time.

**Literature:** FuXi-S2S perturbation module.

---

## Strategy 6: Post-Processing Stacking (Two-Stage Model)

After training the Flow Matcher, train a second lightweight model (Ridge Regression, shallow CNN, or XGBoost) that takes:
- Flow Matcher ensemble mean + spread
- Raw GEOS ensemble mean + spread
- MJO indices, month, lead

...and produces a corrected Week 4 forecast. This corrector learns systematic biases the Flow Matcher hasn't resolved.

**Literature:** NOAA CPC post-processing, 2025 stacking papers, ECMWF AI Weather Quest.

---

## Strategy 7: Additional Conditioning Variables

Add slow-varying predictors that are most useful at Week 3–4 timescales:

| Variable | Source | Why It Helps Week 4 |
|---|---|---|
| **OLR** (Outgoing Longwave Radiation) | ERA5 / NOAA | Raw tropical convection field, richer than RMM indices |
| **QBO** (Quasi-Biennial Oscillation) | Single scalar index | Modulates MJO strength and teleconnections |
| **Niño 3.4 SST anomaly** | Derived from SST | Explicit ENSO state for tropical Pacific modulation |
| **Snow Cover / Sea Ice** | ERA5 | Slow boundary conditions dominating long leads |

These variables don't hurt Weeks 1–2 because they're nearly constant at short timescales.

**Literature:** FuXi-S2S, CPC S2S research, multiple 2025 papers.

---

## Strategy 8: Synthetic Data Augmentation / Ensemble Distillation

Use a generative model (e.g., GenCast) to produce additional training samples for rare/extreme events, augmenting the 16-year training set. Week 4 suffers most from data scarcity because extreme events at long leads are dramatically underrepresented.

**Literature:** 2025 GenCast distillation paper, ECMWF AIFS.

---

## Master Comparison Table

| # | Strategy | Effort | Expected Week 4 Gain | Risk to Weeks 1-2 | Literature Support |
|---|---|---|---|---|---|
| 1 | **Dedicated Output Heads** ✅ | Medium | **High** | **None** | DUNE, Multi-Task Learning |
| 2 | **FiLM / Cross-Attention** | Medium-High | **High** | **None** | ModAFNO 2024, FiLM |
| 3 | **Curriculum Learning** | **Low** | Medium | **None** | FuXi-S2S, standard DL |
| 4 | **MJO Bottleneck Shortcut** | **Low** | Medium-High | **None** | FuXi-S2S |
| 5 | **Flow-Dependent Perturbations** | High | Medium | None | FuXi-S2S |
| 6 | **Post-Processing Stacking** | Medium | Medium-High | **None** | NOAA CPC, 2025 papers |
| 7 | **Extra Conditioning (OLR/QBO)** | **Low-Medium** | **High** | **None** | FuXi-S2S, CPC S2S |
| 8 | **Synthetic Data Augmentation** | High | Medium | Low | 2025 GenCast |

## Recommended Priority Order

1. ✅ **Dedicated Output Heads** — Already implemented
2. **Curriculum Learning** — Easiest next step, training schedule only
3. **MJO Bottleneck Shortcut** — Low effort, high value
4. **Extra Conditioning (OLR, QBO)** — Data pipeline addition
5. **FiLM Conditioning** — Most powerful long-term
6. **Post-Processing Stacking** — After base model matures
7. **Flow-Dependent Perturbations** — Advanced ensemble tuning
8. **Synthetic Data Augmentation** — Requires external generative model
