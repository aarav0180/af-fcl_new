"""
Feature 5: Task-Similarity Adaptive Explore-Theta
==================================================

The paper uses a fixed `flow_explore_theta` hyper-parameter:
    k_explore = (1 - θ) * prob_mean + θ

This mixes prob_mean with a constant θ — a static heuristic.
For EMNIST-LTP (unrelated tasks) θ should be small (aggressive forgetting).
For EMNIST-shuffle (same tasks) θ should be large (preserve knowledge).
But the paper hand-tunes θ per dataset, which doesn't adapt over time.

Fix: Compute θ DYNAMICALLY per training round using cross-task log-likelihood:

    θ*_t = σ( β · E_{x~D^t_k}[ log p_{g'}(h_a(x), y) ] )

where g' is the old task's NF, and σ is the sigmoid function.

Intuition:
  - If current task is SIMILAR to past: current data has HIGH likelihood 
    under old flow → high θ → preserve old knowledge
  - If current task is UNRELATED: current data has LOW likelihood 
    under old flow → low θ → aggressive forgetting

This quantity is the CROSS-ENTROPY between the current task's feature 
distribution and the old NF model — a proper information-theoretic measure
of task similarity. It's computed as a byproduct of the NF forward pass
at no extra cost.
"""

import torch
import torch.nn.functional as F


def compute_adaptive_theta(classifier, last_flow, x, y, num_classes,
                           beta=1.0, default_theta=0.5):
    """
    Compute data-driven explore-theta based on task similarity.
    
    θ* = σ( β · mean( log p_{g'}(h_a(x), y) ) )
    
    where g' is the NF from the previous task and h_a is the current 
    feature extractor.
    
    The mean log-probability measures how well the old flow models the 
    current task's features. Sigmoid bounds the output to [0, 1].
    
    Args:
        classifier:    Current classifier (for feature extraction)
        last_flow:     NF model from the previous task (g')
        x:             Current batch data [N, C, H, W]
        y:             Current batch labels [N]
        num_classes:   Total number of classes
        beta:          Scaling parameter for sigmoid sensitivity
                       Higher β → more sensitive to task similarity
                       (default 1.0)
        default_theta: Fallback theta if no previous flow exists
        
    Returns:
        theta: Adaptive explore-theta in [0, 1]
    """
    if last_flow is None:
        return default_theta

    with torch.no_grad():
        # Extract features from current classifier
        xa = classifier.forward_to_xa(x)
        xa = xa.reshape(xa.shape[0], -1)

        # One-hot encode labels
        y_one_hot = F.one_hot(y, num_classes=num_classes).float()

        # Compute log-likelihood of current features under OLD flow
        # This is the cross-entropy between current task and old model
        log_prob = last_flow.log_prob(inputs=xa, context=y_one_hot)

        # Normalize by feature dimension to get per-dimension log-prob
        # This makes the scale independent of feature dimensionality
        D = xa.shape[1]
        mean_log_prob = log_prob.mean() / D

        # Sigmoid to bound in [0, 1]
        theta = torch.sigmoid(beta * mean_log_prob).item()

    return theta


def compute_explore_forget_weight(theta, prob_mean):
    """
    Compute the explore-forget weight using adaptive theta.
    
    k_explore = (1 - θ*) * prob_mean + θ*
    
    Same formula as original, but θ* is now data-driven instead of fixed.
    
    When θ* is high (similar tasks): weight ≈ 1 (preserve everything)
    When θ* is low (different tasks): weight ≈ prob_mean (selective preservation)
    
    Args:
        theta:     Adaptive theta from compute_adaptive_theta
        prob_mean: Mean probability of generated features in local data
        
    Returns:
        k_explore: Scalar weight for the flow-generated loss
    """
    return (1 - theta) * prob_mean + theta
