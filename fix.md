# Flow Multi v1 Fix Plan

## Current Failure Modes

- The flow transport is trained with pure Gaussian latent noise, but CRPS validation swaps in EOF/LHS teleconnection noise only at inference time. That creates a prior mismatch.
- The multi-flow dataset does not pass the exact initialization year/day into the validation noise path, so the structured sampler can fall back to proxy dates.
- The current sampler leans too hard on teleconnection templates. Pure random noise is already a strong baseline for PR and T2M, so replacing it entirely is hurting calibration.
- NAO and ENSO currently affect the sampler more than the transport network itself, which limits how much those modes can improve CRPS.

## Recommendations

- Keep pure random noise as the base substrate for ensemble generation.
- Add teleconnection structure as a regime-conditioned residual correction instead of replacing the random prior.
- Use the exact initialization date for MJO, NAO, and ENSO lookups in the sampler.
- Move toward direct regime conditioning in the model and variance head with scalar features such as `RMM1`, `RMM2`, MJO amplitude, NAO/AO, and ONI.
- Favor soft mixtures over winner-take-all regime selection when combining random, MJO, NAO, ENSO, and future modes such as PNA.
- Tune CRPS with regime-aware spread calibration after the plumbing is fixed.

## Immediate Work

1. Plumb exact init `year/month/day` through `dataset_flow_multi.py` and the `train_flow_multiv1.py` validation path.
2. Replace the current pure-random versus EOF-only split with hybrid random-plus-regime residual noise in `noise_utils_multi.py`.
3. Keep the hybrid scale configurable so we can tune how strongly regime structure perturbs the random baseline.
4. Verify the updated validation path with a quick compile check and targeted inspection.

## Next Likely Steps

1. Add explicit teleconnection scalar channels to the model conditioning path.
2. Make PR and T2M share correlated regime coefficients rather than fully independent channel-wise draws.
3. Run regime-stratified CRPS diagnostics by MJO amplitude, NAO sign, ENSO state, month, and lead week.
