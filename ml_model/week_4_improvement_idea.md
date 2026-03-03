# Week 4 Improvement Idea: Flow-Dependent Ensemble Spread (Strategy 5)

## The Problem
Currently, our Flow Matching ensemble spread is driven entirely by isotropic Gaussian noise ($x_0 \sim \mathcal{N}(0, I)$). The starting noise is uniform everywhere and does not dynamically adjust based on atmospheric predictability. In reality, subseasonal predictability is highly flow-dependent: strong MJO events should have tight consensus (low variance), while chaotic transition periods should have high uncertainty (high variance).

## The Goal
To predict the optimal noise covariance structure (ensemble variance/spread) dynamically pixel-by-pixel, similarly to how the FuXi model applies flow-dependent perturbations.

## Implementation Strategies

### Approach A: Gaussian Negative Log-Likelihood (The FuXi Method)
Modify the main UNet heads to output 2 channels: `[Mean_Velocity, Log_Variance]`.
- **Loss:** Switch from standard MSE to Gaussian NLL: $\text{Loss} = \frac{(v_{true} - v_{pred})^2}{2\exp(\log(\sigma_{pred}^2))} + \frac{1}{2}\log(\sigma_{pred}^2)$
- **Pros:** Most mathematically sound, network naturally learns to downweight chaotic regions.
- **Cons:** Very unstable training initially, and technically breaks the pure Rectified Flow optimal transport velocity mapping.

### Approach B: Auxiliary Network (Post-Processing)
Leave the Flow Matching core running purely on MSE. Freeze weights, then train a separate standalone network mapping `[Conditioning + $x_{output}$] \rightarrow \text{Variance}`.
- **Pros:** Completely isolates and protects the Flow Matching velocity mathematical stability.
- **Cons:** Requires managing two entirely separate training pipelines and model weight sets.

### Approach C: The Flow-Matching Hybrid Approach (Recommended)
This approach gives the best of both worlds—it maintains pure MSE for the main Flow Matching trajectory while simultaneously training a variance map without destabilizing the core physics.

1.  **Architecture:** Keep the `out_channels=64` intermediate feature map. Give each lead week *two* separate linear heads:
    *   `Mean_Head(64, 1):` Predicts standard velocity field.
    *   `Variance_Head(64, 1):` Predicts spatial uncertainty.
2.  **Gradient Isolation Trick:** Stop gradients from `Variance_Head` from flowing backward into the main UNet body using `.detach()`.
    *   The main UNet trains via pure Flow Matching MSE ($\text{loss}_v = (v_{true} - v_{pred})^2$).
    *   The Variance Head watches the 64-channel feature map and is trained to predict the squared error map: $\text{loss}_{var} = (\sigma^2_{pred} - (v_{true} - v_{pred}\text{.detach()})^2)^2$.
3.  **Inference:**
    Generate the base base precipitation mean via standard ODE solving.
    Retrieve the predicted $\sigma^2(x, y)$ map.
    Instead of using standard $\mathcal{N}(0, 1)$ noise to seed the 10 ensemble members, scale the pixel noise by $\sigma_{pred}$:
    $x_{0[member]} \sim \mathcal{N}(0, 1) \cdot \sigma_{pred}(x, y)$

## Summary
The Hybrid Approach allows us to implement FuXi-style localized ensemble scaling within a single training loop without breaking the fundamental Differential Equation math required for Rectified Flow.
