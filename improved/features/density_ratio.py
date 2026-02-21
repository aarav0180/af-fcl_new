"""
Feature 1: Density Ratio Credibility Estimation
================================================

Replaces the absolute density p_{D^t_k}(u_bar) from Eq. 8 of the paper with
the density RATIO:

    w_i = p_{D^t_k}(u_bar_i) / p_prior(u_bar_i)

where p_prior = N(0, I) is the base distribution of the normalizing flow.

Mathematical justification:
    The NF generates u_bar ~ N(0, I), so absolute density p_{D^t_k}(u_bar)
    is high whenever the local Gaussian overlaps with the prior — regardless
    of whether the sample is actually informative for the current task.
    
    The density ratio corrects for this by measuring how much MORE likely a 
    sample is under the local task distribution compared to the prior.
    This is an importance sampling correction (Sugiyama et al., 2012).

    In log space for diagonal covariance:
    log w_i = 0.5 * ||u_bar||^2 
              - 0.5 * (u_bar - mu)^T Sigma^{-1} (u_bar - mu)
              - 0.5 * sum(log(sigma^2))

    We use the per-dimension geometric mean for numerical stability:
    w_i = exp(mean_over_dims(log_ratio))
"""

import torch
import numpy as np

eps = 1e-30


def sample_from_flow_with_noise(flow, labels, batch_size, num_classes):
    """
    Sample from a normalizing flow AND return the latent noise vectors.
    
    The original sample_from_flow only returns the generated features.
    We need the noise vectors u_bar to compute density ratios in latent space.
    
    Returns:
        flow_xa: Generated features in feature space [batch_size, D]
        label: Class labels (numpy array) [batch_size]
        class_onehot: One-hot encoded labels [batch_size, num_classes]
        noise: Latent noise vectors u_bar ~ N(0,I) [batch_size, D]
    """
    label = np.random.choice(labels, batch_size)
    class_onehot = np.zeros((batch_size, num_classes))
    class_onehot[np.arange(batch_size), label] = 1
    device = next(flow.parameters()).device
    class_onehot = torch.Tensor(class_onehot).to(device)

    # Sample noise from the base distribution N(0, I)
    noise = flow._distribution.sample(batch_size)

    # Get the embedded context 
    embedded_context = flow._embedding_net(class_onehot)

    # Transform noise -> feature space via inverse transform
    flow_xa, _ = flow._transform.inverse(noise, context=embedded_context)
    flow_xa = flow_xa.detach()
    noise = noise.detach()

    return flow_xa, label, class_onehot, noise


def density_ratio_credibility(xa_u, y, prob_mean, flow_noise, flow_label, eps=1e-30):
    """
    Compute density ratio w_i = p_{D^t_k}(u_bar_i) / p_{prior}(u_bar_i)
    for each generated sample's latent vector.

    Both xa_u and flow_noise live in the same latent space (output of g(·)).
    
    Args:
        xa_u:       Latent codes of local real data u = g(h_a(x))  [N, D]
        y:          Labels of local data                            [N]
        prob_mean:  Fallback probability for classes with no local data (scalar)
        flow_noise: Latent noise vectors of generated features      [M, D]
        flow_label: Labels assigned to generated features           [M] (numpy array)
        
    Returns:
        flow_weights: Per-sample credibility weights [M]
    """
    D = xa_u.shape[1]
    device = xa_u.device

    # Convert flow_label to tensor for masking
    if not isinstance(flow_label, torch.Tensor):
        flow_label_t = torch.from_numpy(np.array(flow_label)).long().to(device)
    else:
        flow_label_t = flow_label.long().to(device)

    flow_weights = torch.zeros(flow_noise.shape[0], device=device)

    # log p_prior(u) = -D/2 * log(2pi) - 0.5 * ||u||^2
    # We only need the per-sample part since -D/2 * log(2pi) cancels in the ratio
    log_prior_per_dim = -0.5 * (flow_noise ** 2)  # [M, D]

    flow_label_set = set(flow_label) if not isinstance(flow_label, torch.Tensor) \
                     else set(flow_label.cpu().numpy().tolist())

    for yi in flow_label_set:
        yi = int(yi)
        mask_local = (y == yi)
        mask_flow = (flow_label_t == yi)

        if mask_local.sum() > 0:
            # Fit per-class diagonal Gaussian to local latent codes
            xa_u_yi = xa_u[mask_local]                              # [n_yi, D]
            mu_yi = xa_u_yi.mean(dim=0)                             # [D]
            var_yi = ((xa_u_yi - mu_yi.unsqueeze(0)) ** 2).mean(dim=0) + eps  # [D]

            flow_noise_yi = flow_noise[mask_flow]                   # [m_yi, D]

            # log p_local(u) per dimension = -0.5*log(var) - 0.5*(u-mu)^2/var
            # (ignoring the constant -0.5*log(2pi) which cancels in ratio)
            log_local_per_dim = (
                -0.5 * torch.log(var_yi).unsqueeze(0)
                - 0.5 * (flow_noise_yi - mu_yi.unsqueeze(0)) ** 2 / var_yi.unsqueeze(0)
            )  # [m_yi, D]

            log_prior_yi_per_dim = log_prior_per_dim[mask_flow]     # [m_yi, D]

            # Log density ratio per dimension
            log_ratio_per_dim = log_local_per_dim - log_prior_yi_per_dim  # [m_yi, D]

            # Geometric mean over dimensions for numerical stability
            # w_i = exp(mean_d(log_ratio_d))
            log_ratio_mean = log_ratio_per_dim.mean(dim=1)          # [m_yi]

            # Clamp for stability and exponentiate
            weights = torch.exp(log_ratio_mean.clamp(min=-10, max=10))
            flow_weights[mask_flow] = weights
        else:
            # No local data for this class — use fallback
            flow_weights[mask_flow] = prob_mean

    return flow_weights
