"""
Feature 6: Sinkhorn Divergence Feature Distillation (replaces Eq. 6)
====================================================================

The paper's KD loss (Eq. 6) is isotropic MSE:
    L_KD = (1/n) * Σ_i ||h_a(x_i) - h'_a(x_i)||^2

This treats all 512 feature dimensions identically and forces per-sample
feature matching. In continual learning, the model needs to REORGANIZE
its feature representation for new tasks while preserving the distributional
shape needed by the NF.

Fix: Replace MSE with Sinkhorn divergence (entropic-regularized Wasserstein):
    L_KD^W = W_ε( P_current, P_old )

where P_current = {h_a(x_i)} and P_old = {h'_a(x_i)}.

Key difference:
  - MSE minimizes per-sample pointwise distance (features must stay close 
    for EVERY individual sample)
  - Sinkhorn minimizes DISTRIBUTIONAL distance (overall feature distribution 
    must be preserved, but individual features can shift as long as the 
    distribution's shape is maintained)

This allows the model to reorganize features (good for new tasks) while
preserving the collective distributional shape (good for NF stability).

The Sinkhorn divergence is unbiased and metrizes weak convergence:
    S_ε(P, Q) = W_ε(P, Q) - 0.5 * W_ε(P, P) - 0.5 * W_ε(Q, Q)

Implementation uses log-domain stabilized Sinkhorn iterations for numerical
stability in high dimensions.
"""

import torch


def _log_sinkhorn_iterations(log_a, log_b, M, reg, max_iter):
    """
    Stabilized log-domain Sinkhorn iterations.
    
    Solves the entropic OT problem:
        min_P <P, M> + reg * KL(P || a ⊗ b)
    
    Args:
        log_a: Log of source marginal [N]
        log_b: Log of target marginal [M]
        M:     Cost matrix [N, M_]
        reg:   Regularization strength (epsilon)
        max_iter: Number of Sinkhorn iterations
        
    Returns:
        log_P: Log of transport plan [N, M_]
    """
    # Initialize dual variables
    f = torch.zeros_like(log_a)  # [N]
    g = torch.zeros_like(log_b)  # [M_]

    for _ in range(max_iter):
        # f update: f_i = - reg * logsumexp_j( (-M_ij + g_j) / reg ) + f_i
        f = -reg * torch.logsumexp((-M + g.unsqueeze(0)) / reg, dim=1) + f
        # Incorporate marginal constraint
        f = f + log_a * reg

        # g update
        g = -reg * torch.logsumexp((-M + f.unsqueeze(1)) / reg, dim=0) + g
        g = g + log_b * reg

    # Transport plan in log domain
    log_P = (-M + f.unsqueeze(1) + g.unsqueeze(0)) / reg
    return log_P


def _sinkhorn_cost(X, Y, reg=0.1, max_iter=30):
    """
    Compute the Sinkhorn (entropic OT) cost between two point clouds.
    
    W_ε(X, Y) = <P*, C> where P* solves the regularized OT problem.
    
    Args:
        X: Point cloud 1 [N, D]
        Y: Point cloud 2 [M, D]
        reg: Entropic regularization strength
        max_iter: Number of Sinkhorn iterations
        
    Returns:
        cost: Scalar Sinkhorn cost
    """
    N = X.shape[0]
    M = Y.shape[0]

    # Squared Euclidean cost matrix
    C = torch.cdist(X, Y, p=2.0).pow(2)  # [N, M]

    # Normalize cost for stability (important in high dimensions)
    C = C / (C.max().detach() + 1e-8)

    # Uniform marginals in log domain
    log_a = torch.full((N,), -torch.log(torch.tensor(float(N))),
                       device=X.device)
    log_b = torch.full((M,), -torch.log(torch.tensor(float(M))),
                       device=X.device)

    # Sinkhorn iterations
    f = torch.zeros(N, device=X.device)
    g = torch.zeros(M, device=X.device)

    for _ in range(max_iter):
        # Update f (row normalization)
        f = -reg * torch.logsumexp((-C + g.unsqueeze(0)) / reg, dim=1)
        # Update g (column normalization)
        g = -reg * torch.logsumexp((-C + f.unsqueeze(1)) / reg, dim=0)

    # Compute OT cost
    log_P = (-C + f.unsqueeze(1) + g.unsqueeze(0)) / reg
    P = torch.exp(log_P)
    # Normalize P to be a valid transport plan
    P = P / (P.sum() + 1e-8)
    cost = (P * C).sum()

    return cost


def sinkhorn_divergence(X, Y, reg=0.1, max_iter=30):
    """
    Compute the Sinkhorn divergence (unbiased, metrizes weak convergence):
    
        S_ε(X, Y) = W_ε(X, Y) - 0.5 * W_ε(X, X) - 0.5 * W_ε(Y, Y)
    
    The self-terms W_ε(X,X) and W_ε(Y,Y) remove the entropic bias,
    making S_ε a proper divergence that vanishes iff X =_d Y.
    
    Args:
        X: Current features h_a(x) [N, D]
        Y: Old features h'_a(x) [N, D] (or EMA features)
        reg: Entropic regularization (ε). 
             Smaller → closer to true Wasserstein but less stable.
             Default 0.1 works well for 512-dim features.
        max_iter: Sinkhorn iterations (30 is usually sufficient)
        
    Returns:
        divergence: Scalar Sinkhorn divergence (differentiable w.r.t. X)
    """
    W_xy = _sinkhorn_cost(X, Y, reg=reg, max_iter=max_iter)
    W_xx = _sinkhorn_cost(X, X, reg=reg, max_iter=max_iter)

    # Y is detached (old model) so W_yy has no gradient — just a constant
    with torch.no_grad():
        W_yy = _sinkhorn_cost(Y, Y, reg=reg, max_iter=max_iter)

    divergence = W_xy - 0.5 * W_xx - 0.5 * W_yy
    return divergence


def sinkhorn_feature_distillation(xa_current, xa_old, k_kd_coeff,
                                  reg=0.1, max_iter=30):
    """
    Compute feature distillation loss using Sinkhorn divergence.
    
    Replacement for the MSE-based Eq. 6:
        L_KD^W = k * S_ε({h_a(x_i)}, {h'_a(x_i)})
    
    Args:
        xa_current:  Current classifier features h_a(x) [N, D]
        xa_old:      Old classifier features h'_a(x) [N, D] (detached)
        k_kd_coeff:  KD loss weight coefficient
        reg:         Sinkhorn regularization parameter
        max_iter:    Number of Sinkhorn iterations
        
    Returns:
        kd_loss: Scalar Sinkhorn distillation loss
    """
    xa_old = xa_old.detach()
    kd_loss = k_kd_coeff * sinkhorn_divergence(
        xa_current, xa_old, reg=reg, max_iter=max_iter)
    return kd_loss
