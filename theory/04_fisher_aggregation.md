# Feature 4: Fisher Information Weighted Aggregation

**CLI flag**: `--use_fisher_aggregation`
**File**: `improved/features/fisher_aggregation.py`

---

## What the Original Paper Does

After local training, the server aggregates all parameters (classifier + NF) using Federated Averaging:

$$
\theta_{\text{server}} = \sum_{k=1}^{K} \frac{n_k}{\sum_j n_j} \theta_k
$$

This is a simple weighted average by data size.

---

## What is Wrong

FedAvg works reasonably for classifier parameters (where the loss landscape is relatively convex near the minimum). But for normalizing flow parameters, averaging is problematic.

**Why NFs are different from classifiers:**

A normalizing flow defines a **bijection** (invertible mapping). Two NFs g_1 and g_2 trained on different data may learn very different invertible mappings that happen to model similar distributions. Their parameter-space representations can be completely different even when their functional behavior is similar.

**Concrete example**: Consider two affine coupling layers. NF_1 might learn to shift class-A features up and class-B features down. NF_2 might learn the opposite encoding. Averaging their parameters produces a mapping that shifts nothing --- a garbled transformation that models no distribution well.

This is analogous to the "permutation symmetry" problem in neural networks, but worse for NFs because they must remain invertible.

---

## The Fix: Fisher Information Weighted Aggregation

Weight each parameter dimension by how much the local data constrains it:

$$
\theta_{\text{server}} = \left(\sum_{k=1}^{K} F_k\right)^{-1} \sum_{k=1}^{K} F_k \theta_k
$$

where F_k is the diagonal Fisher Information Matrix for client k.

---

## The Fisher Information Matrix

### Definition

For each NF parameter theta^{(j)}, the Fisher information measures how sensitive the log-likelihood is to that parameter:

$$
F_k^{(j)} = \frac{1}{n_k} \sum_{i=1}^{n_k} \left(\frac{\partial \log p_{g_k}(h_a(x_i), y_i)}{\partial \theta^{(j)}}\right)^2
$$

### Interpretation

| Fisher value | Meaning | Aggregation effect |
|-------------|---------|-------------------|
| Large F_k^{(j)} | Parameter tightly constrained by client k's data | Client k's value dominates |
| Small F_k^{(j)} | Parameter barely affects likelihood for client k | Other clients determine it |

This is information-theoretically optimal: parameters are weighted by how informative each client is about them.

---

## Bayesian Justification

The Fisher-weighted average is the MAP estimate under a product-of-Gaussians model.

Each client provides an approximate posterior for the NF parameters (Laplace approximation):

$$
p(\theta | D_k) \approx \mathcal{N}(\theta_k, F_k^{-1})
$$

The product of all client posteriors:

$$
p(\theta | D_1, \ldots, D_K) \propto \prod_{k=1}^{K} \mathcal{N}(\theta; \theta_k, F_k^{-1})
$$

The mode (MAP estimate) of this product Gaussian is:

$$
\theta_{\text{MAP}} = \left(\sum_{k=1}^{K} F_k\right)^{-1} \sum_{k=1}^{K} F_k \theta_k
$$

This is exactly the Fisher-weighted average. It is the **correct Bayesian aggregation formula** under Laplace approximation.

---

## Why This is Especially Good for NFs

For normalizing flows, log p_g(z, y) is **exactly computable** via the change-of-variables formula (Eq. 3). This means:

1. The Fisher is computed from the **exact** log-likelihood, not an approximation.
2. The gradients come from the same backward pass used for training --- no extra computation.
3. For classifiers, the Fisher uses cross-entropy loss (an approximation to the true data likelihood). For NFs, it uses the true generative log-probability.

---

## Scope of Application

| Component | Aggregation method |
|-----------|-------------------|
| Classifier parameters | Standard FedAvg (robust to averaging) |
| NF parameters | **Fisher-weighted** aggregation |

Only NF parameters use Fisher weighting. Classifier parameters are still aggregated with standard FedAvg because they are more robust to parameter averaging (shared representation learning).

---

## Efficient Batch Approximation

The exact per-sample Fisher requires n_k backward passes (one per sample). The batch approximation uses one backward pass per batch:

$$
\tilde{F}_k^{(j)} \approx \frac{1}{B} \sum_{b=1}^{B} \left(\frac{\partial \mathcal{L}_b}{\partial \theta^{(j)}}\right)^2, \quad \mathcal{L}_b = -\frac{1}{|\text{batch}_b|} \sum_{i \in \text{batch}_b} \log p_{g_k}(z_i, y_i)
$$

This runs B times faster while providing a reasonable lower-bound approximation to the true diagonal Fisher.
