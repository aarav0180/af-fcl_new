# Feature 3: EMA Feature Extractor

**CLI flag**: `--use_ema_extractor`
**Hyperparameter**: `--ema_alpha` (default 0.999)
**File**: `improved/features/ema_extractor.py`

---

## What the Original Paper Does

The NF is trained to model the distribution of features h_a(x) conditioned on y:

$$
\mathcal{L}_{\text{NF}} = -\frac{1}{n} \sum_{i=1}^{n} \log p_z(h_a(x_i), y_i)
$$

The feature extractor h_a and the NF g are trained simultaneously.

---

## What is Wrong

The NF is trying to learn the distribution p(h_a(x), y), but h_a changes at every training step. This is a **moving target problem**.

At step tau, the NF learns to model p(h_a^{(tau)}(x), y). At step tau+1, the classifier updates h_a, and the NF's learned distribution is now stale. The NF is always chasing a target that keeps moving.

The KD loss (Eq. 6) partially constrains h_a but creates a deadlock:

| KD strength | h_a behavior | NF behavior | Problem |
|-------------|-------------|-------------|---------|
| Strong (large k_KD) | Cannot adapt | Stable target | Poor new-task accuracy |
| Weak (small k_KD) | Drifts freely | Target moves | NF becomes invalid |

There is no setting of k_KD that solves both problems simultaneously.

---

## The Fix: Exponential Moving Average

Maintain an EMA copy of the feature extractor that changes slowly:

$$
h_a^{\text{EMA}}(\tau) = \alpha \cdot h_a^{\text{EMA}}(\tau - 1) + (1 - \alpha) \cdot h_a(\tau)
$$

where alpha is in [0.99, 0.999].

Train the NF on features from the EMA model instead of the live classifier:

$$
\mathcal{L}_{\text{NF}}^{\text{EMA}} = -\frac{1}{n} \sum_{i=1}^{n} \log p_z(h_a^{\text{EMA}}(x_i), y_i)
$$

---

## Why EMA Works

### What EMA actually computes

Unrolling the recursion:

$$
h_a^{\text{EMA}}(\tau) = (1 - \alpha) \sum_{s=0}^{\tau-1} \alpha^s \cdot h_a^{(\tau - s)}
$$

This is a weighted average of **all past iterates**, with exponentially decaying weights. Recent iterates get the most weight, but old iterates still contribute.

### Variance reduction

If h_a^{(tau)} = h_a* + epsilon_tau where epsilon_tau is zero-mean noise with variance sigma^2:

$$
\text{Var}[h_a^{\text{EMA}}(\tau)] = \frac{1 - \alpha}{1 + \alpha} \cdot \sigma^2
$$

For alpha = 0.999:

$$
\text{Var}[h_a^{\text{EMA}}] \approx 0.0005 \cdot \sigma^2
$$

That is a **2000x reduction** in variance. The EMA model changes smoothly and slowly.

### Decoupling two time scales

With EMA, we get a two-timescale system:
- **Fast**: The live classifier h_a adapts quickly to the new task (free to change).
- **Slow**: The EMA h_a^{EMA} changes gradually, giving the NF a stable target.

This is the same principle as Mean Teacher (Tarvainen and Valpola, 2017). The NF is guaranteed to have a Lyapunov-stable target.

---

## Modified KD with EMA

Optionally, the feature KD loss can also use EMA instead of the frozen old model:

$$
\mathcal{L}_{\text{KD}}^{\text{EMA}} = \frac{1}{n} \sum_{i=1}^{n} \| h_a(x_i) - h_a^{\text{EMA}}(x_i) \|^2
$$

This replaces the frozen h_a' (which becomes increasingly stale over training) with a continuously adapting smooth reference.

---

## When EMA is Re-initialized

At each **task boundary** (when new task data arrives), the EMA is re-initialized from the current classifier state. This ensures the EMA starts fresh for each task rather than carrying stale information from the previous task's feature space.
