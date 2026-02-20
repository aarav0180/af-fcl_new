# Feature 5: Adaptive Explore-Theta

**CLI flag**: `--use_adaptive_theta`
**Hyperparameter**: `--adaptive_theta_beta` (default 1.0)
**File**: `improved/features/adaptive_theta.py`

---

## What the Original Paper Does

The explore-forget weight for generated-feature classification loss is:

$$
k_{\text{explore}} = (1 - \theta) \cdot \bar{p} + \theta
$$

where:
- p_bar = mean credibility of generated features.
- theta is a **fixed** hyperparameter, hand-tuned per dataset.

The paper uses these values:

| Dataset | theta |
|---------|-------|
| EMNIST-Letters (unrelated tasks) | 0.0 |
| EMNIST-Shuffle (similar tasks) | 0.5 |
| MNIST-SVHN-FASHION | 0.0 |
| CIFAR100 | 0.1 |

---

## What is Wrong

The optimal theta depends on **how similar the current task is to past tasks**. This varies:
- **Across tasks**: Early tasks may be similar to each other; later ones may differ completely.
- **Across clients**: Different clients see different class subsets, so the same task may be "similar" for one client and "different" for another.
- **Over training rounds**: As the model improves, the relationship between old and new representations changes.

A single fixed theta per dataset cannot adapt to any of this. The only way to set it is trial-and-error hyperparameter search, which does not generalize.

**When theta is wrong**:
- theta too large (e.g., 0.5 when tasks are unrelated): the model preserves useless old knowledge, wasting capacity.
- theta too small (e.g., 0.0 when tasks are similar): the model aggressively forgets useful knowledge.

---

## The Fix: Data-Driven Theta

Compute theta dynamically using the cross-task NF log-likelihood:

$$
\theta^*_t = \sigma\left(\beta \cdot \frac{1}{n \cdot D} \sum_{i=1}^{n} \log p_{g'}(h_a(x_i), y_i)\right)
$$

where:
- g' is the NF from the previous task (frozen).
- sigma(.) is the sigmoid function: sigma(x) = 1/(1 + e^{-x}).
- beta is a scaling parameter (sensitivity control).
- D is the feature dimension (for scale normalization).

---

## Step-by-Step Derivation

### Step 1: Measure task similarity

The quantity:

$$
S = \frac{1}{n} \sum_{i=1}^{n} \log p_{g'}(h_a(x_i), y_i)
$$

is the average log-likelihood of the current task's features under the **old** flow g'. This is a proper information-theoretic measure of task similarity:

- S is high (close to 0): current features are well-modeled by old flow, meaning tasks are **similar**.
- S is very negative: current features are unlikely under old flow, meaning tasks are **different**.

### Step 2: Normalize by dimension

Divide by D to get a per-dimension log-probability, making the score independent of feature dimensionality:

$$
\bar{S} = \frac{S}{D} = \frac{1}{n \cdot D} \sum_{i=1}^{n} \log p_{g'}(h_a(x_i), y_i)
$$

### Step 3: Map to [0, 1] with sigmoid

$$
\theta^* = \sigma(\beta \cdot \bar{S}) = \frac{1}{1 + e^{-\beta \bar{S}}}
$$

The sigmoid ensures theta is bounded in [0, 1].

---

## Behavior

| Situation | S_bar | theta* | k_explore | Effect |
|-----------|-------|--------|-----------|--------|
| Tasks very similar | High (near 0) | Close to 1 | ~1 | Preserve all old knowledge |
| Tasks moderately similar | Medium | ~0.5 | ~0.5 * p_bar + 0.5 | Balanced |
| Tasks completely different | Very negative | Close to 0 | ~p_bar | Selective preservation (credibility only) |

---

## Why This is Free

The computation uses the same NF forward pass that is already done during training. The old flow g' is already available (kept as `last_flow`). Computing log p_{g'}(h_a(x), y) requires one forward pass through g', which is cheap compared to the rest of training.

---

## Role of Beta

Beta controls the sigmoid's sensitivity:
- Large beta: sharp transition between "preserve" and "forget" regimes.
- Small beta: gradual transition.
- Beta = 1.0 (default): moderate sensitivity. Adjust based on the expected range of S_bar for the dataset.
