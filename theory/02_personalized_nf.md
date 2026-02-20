# Feature 2: Personalized NF with KL Regularization

**CLI flag**: `--use_personalized_nf`
**Hyperparameter**: `--nf_kl_lambda` (default 0.1)
**File**: `improved/features/personalized_nf.py`

---

## What the Original Paper Does

A single global NF is shared by all clients and serves two roles:
1. **Generator**: produces replay features for all clients.
2. **Credibility estimator**: provides the latent space where per-class Gaussians are fit (Eq. 7-8).

The global NF is obtained by FedAvg of all clients' NF parameters.

---

## What is Wrong

These two roles **conflict**.

For **generation**, the NF should model the average feature distribution across all clients (so it can produce representative replay features for everyone).

For **credibility estimation**, the NF should model the **local** client's feature distribution accurately (so the per-class Gaussians in latent space are tight and discriminative).

The global NF is a compromise. The per-class Gaussians in its latent space are broader than they should be (diluted by other clients' data), making credibility scores less discriminative.

**Concrete example**: Client A has classes {dog, cat}. Client B has classes {car, truck}. The global NF tries to model all four classes. When Client A uses it for credibility estimation, the latent space includes structure for car/truck that distorts the dog/cat Gaussians.

---

## The Fix: Personal NF with KL Regularization

Each client k trains a **personal NF** g_k that is regularized to stay close to the server NF:

$$
\mathcal{L}_{\text{NF}}^{k} = \underbrace{\mathcal{L}_{\text{NF}}(g_k; D_k^t, G_z)}_{\text{standard NF loss (Eq. 5)}} + \underbrace{\lambda_{\text{KL}} \cdot D_{\text{KL}}(g_k \| g_{\text{server}})}_{\text{regularization to global}}
$$

---

## Exact KL Between Normalizing Flows

For normalizing flows, KL divergence is **exactly computable** (not a variational bound or approximation):

$$
D_{\text{KL}}(g_k \| g_{\text{server}}) = \mathbb{E}_{z \sim p_{g_k}}\left[\log p_{g_k}(z | y) - \log p_{g_{\text{server}}}(z | y)\right]
$$

Both log p_{g_k} and log p_{g_server} are computed exactly using the NF change-of-variables formula (Eq. 3). In practice, we estimate the expectation using the local data batch:

$$
D_{\text{KL}}(g_k \| g_{\text{server}}) \approx \frac{1}{n} \sum_{i=1}^{n} \left[ \log p_{g_k}(h_a(x_i), y_i) - \log p_{g_{\text{server}}}(h_a(x_i), y_i) \right]
$$

This is exact up to Monte Carlo estimation error. Both log-probabilities involve computing the full NF forward pass (transform + Jacobian determinant), which is already done during training.

---

## How the Two NFs Are Used

| Task | Original | Improved |
|------|----------|----------|
| Replay generation | Global NF g_server | Global NF g_server (unchanged) |
| Credibility estimation (Eq. 8) | Global NF g_server | **Personal NF g_k** |

The global NF still handles generation (it represents all clients). The personal NF handles credibility (it represents the local data accurately).

---

## Why KL Regularization

Without regularization, the personal NF could overfit to the current task's local data and lose information about past tasks. The KL term:

- Keeps g_k close to g_server (which contains knowledge from all clients and tasks).
- Acts like an elastic constraint: g_k can deviate from g_server where local data provides evidence, but snaps back where there is no local signal.
- Lambda_KL controls the trade-off: larger lambda = closer to global (more stable, less personalized), smaller lambda = more personalized (better credibility, risk of overfitting).
