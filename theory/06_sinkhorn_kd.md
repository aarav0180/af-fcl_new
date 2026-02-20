# Feature 6: Sinkhorn Divergence Feature Distillation

**CLI flag**: `--use_sinkhorn_kd`
**Hyperparameters**: `--sinkhorn_reg` (default 0.1), `--sinkhorn_iters` (default 30)
**File**: `improved/features/sinkhorn_kd.py`

---

## What the Original Paper Does

Feature knowledge distillation (Eq. 6) uses pointwise MSE:

$$
\mathcal{L}_{\text{KD}} = \frac{1}{n} \sum_{i=1}^{n} \| h_a(x_i) - h_a'(x_i) \|^2
$$

For each sample x_i, the current features h_a(x_i) are forced to stay close to the old features h_a'(x_i).

---

## What is Wrong

MSE enforces **pointwise** matching: each individual feature vector must stay near its old position. This is too rigid for continual learning.

**The core conflict**: The model needs to reorganize its feature representation for new tasks. New classes may require new directions in feature space. But MSE prevents any reorganization, even reorganization that would preserve the distribution's shape.

**Example**: Suppose all 512-dimensional feature vectors rotate by a small angle (a unitary transformation). The distribution shape is perfectly preserved. The NF would be equally happy. But MSE incurs a large penalty because every point moved.

**What the NF actually needs**: The NF models the **distribution** of features, not individual feature positions. It needs the distribution shape to be preserved, not individual points.

---

## The Fix: Sinkhorn Divergence

Replace MSE with the Sinkhorn divergence, which measures **distributional** distance instead of pointwise distance:

$$
\mathcal{L}_{\text{KD}}^W = S_\varepsilon\left(\{h_a(x_i)\}_{i=1}^n, \; \{h_a'(x_i)\}_{i=1}^n\right)
$$

---

## Optimal Transport Background

### Wasserstein Distance

The 2-Wasserstein distance between two distributions P and Q is:

$$
W_2(P, Q) = \inf_{\gamma \in \Pi(P, Q)} \left(\int \|x - y\|^2 \; d\gamma(x, y)\right)^{1/2}
$$

where Pi(P, Q) is the set of all joint distributions with marginals P and Q.

**Think of it as**: P and Q are piles of dirt. The Wasserstein distance is the minimum total work (mass times distance) to reshape pile P into pile Q. Unlike MSE, it finds the **optimal assignment** between points rather than forcing each point to stay in place.

### Why not use Wasserstein directly?

Computing the exact Wasserstein distance requires solving a linear program, which is O(n^3). For n = 64 (batch size) in D = 512 dimensions, this is too slow for every training iteration.

---

## Entropic Regularization

Add an entropy term to make the optimization problem smooth and fast:

$$
W_\varepsilon(P, Q) = \inf_{\gamma \in \Pi(P, Q)} \left\{ \int \|x - y\|^2 \; d\gamma + \varepsilon \cdot \text{KL}(\gamma \| P \otimes Q) \right\}
$$

The KL term makes the problem strictly convex, solvable via the Sinkhorn algorithm in O(n^2) with just matrix-vector operations.

---

## The Sinkhorn Algorithm

Given point clouds X = {x_i}_{i=1}^N and Y = {y_j}_{j=1}^M:

**Step 1**: Compute the cost matrix C_{ij} = ||x_i - y_j||^2 and normalize C_bar = C / max(C).

**Step 2**: Initialize dual variables f = 0 (length N), g = 0 (length M).

**Step 3**: Iterate (log-domain for numerical stability):

$$
f_i \leftarrow -\varepsilon \cdot \text{logsumexp}_j\left(\frac{-\bar{C}_{ij} + g_j}{\varepsilon}\right)
$$

$$
g_j \leftarrow -\varepsilon \cdot \text{logsumexp}_i\left(\frac{-\bar{C}_{ij} + f_i}{\varepsilon}\right)
$$

Repeat for 30 iterations (converges quickly).

**Step 4**: Compute transport plan: P_{ij} proportional to exp((-C_bar_{ij} + f_i + g_j) / epsilon).

**Step 5**: Compute cost: W_epsilon = sum_{ij} P_{ij} * C_bar_{ij}.

---

## Sinkhorn Divergence (Unbiased)

The entropic regularization introduces a bias: W_epsilon(P, P) > 0 even when P = Q. Correct this:

$$
S_\varepsilon(P, Q) = W_\varepsilon(P, Q) - \frac{1}{2}W_\varepsilon(P, P) - \frac{1}{2}W_\varepsilon(Q, Q)
$$

Properties:
- S_epsilon(P, P) = 0 (unbiased, vanishes for identical distributions).
- S_epsilon(P, Q) >= 0 (non-negative).
- Metrizes weak convergence: S_epsilon(P_n, Q) -> 0 if and only if P_n converges to Q in distribution.
- Differentiable with respect to P (so it can be backpropagated through).

---

## Comparison: MSE vs Sinkhorn

| Property | MSE | Sinkhorn |
|----------|-----|----------|
| What it constrains | Each point stays in place | Distribution shape preserved |
| Feature reorganization | Forbidden | Allowed (if distribution is preserved) |
| Sensitivity to permutations | Very sensitive | Invariant (finds optimal matching) |
| Computation | O(n) | O(n^2 * iters) |
| What the NF needs | Does not match NF's need | Matches exactly |

**MSE**: "Every feature must stay where it was."
**Sinkhorn**: "The collection of features must form the same shape."

The Sinkhorn divergence allows individual features to move, adapt, and reorganize for the new task, as long as the overall distributional geometry is preserved --- which is exactly what the NF requires for stable generative replay.

---

## Hyperparameters

- **epsilon** (sinkhorn_reg, default 0.1): Controls the smoothness of the transport plan. Smaller epsilon gives a plan closer to the true Wasserstein distance but is less numerically stable. For D=512 features, epsilon=0.1 balances accuracy and stability well.
- **max_iter** (sinkhorn_iters, default 30): Number of Sinkhorn iterations. 30 is typically sufficient for convergence. More iterations give a more accurate transport plan but take longer.
