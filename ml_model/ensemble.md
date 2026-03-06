# Ensemble Generation Strategies for Flow-v5

Given the architecture of Flow-v5 (EOF-structured noise + variance head + flow matching), here are five prioritized ensemble generation strategies ordered by impact and implementation cost. These aim to improve both the Continuous Ranked Probability Score (CRPS) and Ensemble Mean Root Mean Square Error (RMSE).

---

## 1. Antithetic Noise Sampling (Immediate Win)
**Impact:** Medium (CRPS), High (RMSE) | **Cost:** Zero | **Timeline:** Immediate

**Concept:** 
For every initial noise draw $z$ sampled from the EOF structure, automatically include its exact mirror $-z$ in the ensemble.

**Why it works:** 
By forcing the ensemble to be symmetric, the expected value $E[f(z) + f(-z)] / 2$ has significantly lower variance than taking random, independent draws of $z$. The random structural biases of the noise perfectly cancel out in the ensemble mean, leading to a direct and mathematically guaranteed drop in RMSE without any model retraining.

---

## 2. Temperature-Scaled Noise Calibration (Fast Calibration)
**Impact:** High (CRPS), Low (RMSE) | **Cost:** Zero | **Timeline:** Immediate

**Concept:** 
Introduce a global temperature parameter $\tau$ that scales the amplitude of the initial noise: $z = \tau \cdot z_{eof}$. Calibrate $\tau$ via a grid search (e.g., $0.5 \to 2.0$) on a held-out validation set.

**Why it works:** 
While the variance head predicts the *relative* spatial spread (e.g., mapping more variance to the ITCZ), the model's overall magnitude of spread might be systematically over- or under-dispersed. Tuning a single $\tau$ scalar forces the ensemble spread to perfectly match the actual observed error distribution, immediately minimizing the CRPS penalty.

---

## 3. Variance-Head Guided Heteroscedastic Noise Injection
**Impact:** High (CRPS), Medium (RMSE) | **Cost:** Low | **Timeline:** Near-term

**Concept:** 
Upgrade the deterministic ODE solver (Euler) to a Stochastic Differential Equation (SDE) solver. Use the spatial scalar map predicted by the Variance Head to physically modulate the width of stochastic perturbations injected at each integration step.

**Why it works:** 
Currently, the Variance Head is only used to isolate the loss gradient. Injecting noise dynamically means that regions the model explicitly knows are uncertain (e.g., storm tracks) receive wider stochastic "kicks" during integration, creating a physically aware spread rather than relying solely on the $t=0$ initial state structure. The noise decays as $t \to 1$ to preserve the target.

---

## 4. CRPS as an Auxiliary Training Loss
**Impact:** High (CRPS), Medium (RMSE) | **Cost:** Medium | **Timeline:** Next Retrain

**Concept:** 
Implement a differentiable Energy Score (a multivariate generalization of CRPS) and add it to the joint loss function alongside the MSE and Variance Head losses.

**Why it works:** 
The model currently learns expected values (via Velocity MSE) and residual magnitude (via Variance Head). Directly penalizing the full ensemble spread via an Energy Score forces the UNet weights to holistically calibrate the diversity of its predictions against the target manifold. Since calculating pair-wise differences is memory intensive, it can be triggered every $K$ steps.

---

## 5. Conditioning Perturbation (Physics-Informed)
**Impact:** Medium (CRPS), Low (RMSE) | **Cost:** Low | **Timeline:** Inference Variant

**Concept:** 
Add small stochastic perturbations to the GEOS forecast condition channels and the MJO phase embeddings before the ODE solve.

**Why it works:** 
In Numerical Weather Prediction (NWP), ensemble spread comes from both stochastic physics (our noise $z$) and Initial Condition (IC) uncertainty. Perturbing the GEOS inputs explicitly models the uncertainty in the underlying driving forecast itself. 

---

### Implementation Roadmap

1. **Immediate:** Add **Antithetic Sampling** to the inference scripts. This costs nothing and directly lowers RMSE.
2. **Immediate:** Run a **Temperature Calibration** loop on validation to find the optimal $\tau$.
3. **Next Week:** Implement the **Heteroscedastic SDE** injection to actively utilize the Variance Head during sampling.
4. **Next Training Run:** Integrate the **CRPS Auxiliary Loss** for the absolute best mathematically calibrated performance.
