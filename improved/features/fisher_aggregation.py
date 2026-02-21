"""
Feature 4: Fisher Information Weighted FedAvg for NF Aggregation
================================================================

Plain FedAvg for NF: θ_server = Σ_k (n_k / Σn) * θ_k^NF

This destroys NF parameter semantics: two clients learning the same
distribution from different init may have NF parameters that, when averaged,
produce a meaningless transformation.

Fix: Weight each parameter dimension by how much the local data constrains it,
using the diagonal Fisher Information Matrix (FIM):

    θ_server = (Σ_k F_k)^{-1} * Σ_k F_k * θ_k

where F_k^{(j)} = (1/n_k) * Σ_{x_i} (∂ log p_{g_k}(x_i, y_i) / ∂θ^{(j)})^2

For NF models, log p is EXACTLY computable (Eq. 3/4), so the Fisher
is available from the backward pass at no extra cost.

The Fisher-weighted average is the MAP estimate under a Gaussian prior —
the correct Bayesian aggregation formula.
"""

import torch
import torch.nn.functional as F


def compute_diagonal_fisher(model, dataloader, device, num_classes, 
                            max_batches=10):
    """
    Compute the diagonal Fisher Information Matrix for the NF parameters
    using local client data.
    
    F^{(j)} = (1/N) * Σ_i (∂ log p_g(h_a(x_i), y_i) / ∂θ^{(j)})^2
    
    This measures how "informative" each NF parameter is for the local data:
    - Large F → parameter tightly constrained by data → should dominate in avg
    - Small F → parameter barely affected by data → let other clients determine
    
    Args:
        model:       PreciseModel instance (needs .classifier and .flow)
        dataloader:  DataLoader for local training data
        device:      torch device
        num_classes: Total number of classes
        max_batches: Limit on batches for efficiency
        
    Returns:
        fisher_diag: Dict[param_name -> Tensor] with diagonal Fisher per NF param
    """
    fisher_diag = {}

    # Initialize Fisher to zeros
    for name, param in model.flow.named_parameters():
        fisher_diag['flow.' + name] = torch.zeros_like(param.data)

    model.classifier.eval()
    model.flow.eval()

    n_samples = 0
    for batch_idx, (x, y) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)
        batch_size = x.shape[0]

        with torch.no_grad():
            xa = model.classifier.forward_to_xa(x)
            xa = xa.reshape(xa.shape[0], -1)

        y_one_hot = F.one_hot(y, num_classes=num_classes).float()

        # Compute log p_g(xa, y) — the NF training objective  
        log_prob = model.flow.log_prob(inputs=xa.detach(), context=y_one_hot)

        # Accumulate per-sample Fisher: E[(∂ log p / ∂θ)^2]
        for i in range(batch_size):
            model.flow.zero_grad()
            log_prob[i].backward(retain_graph=(i < batch_size - 1))

            for name, param in model.flow.named_parameters():
                if param.grad is not None:
                    fisher_diag['flow.' + name] += param.grad.data ** 2

            n_samples += 1

    # Normalize by number of samples
    if n_samples > 0:
        for name in fisher_diag:
            fisher_diag[name] /= n_samples

    # Add small constant for numerical stability (prevents division by zero)
    for name in fisher_diag:
        fisher_diag[name] += 1e-8

    return fisher_diag


def compute_diagonal_fisher_efficient(model, dataloader, device, num_classes,
                                       max_batches=10):
    """
    Efficient approximation: use the batch-level gradient squared instead of
    per-sample. This is a lower-bound approximation of the true Fisher but
    runs batch_size times faster.
    
    F^{(j)} ≈ (1/B) * Σ_b ( (∂ L_b / ∂θ^{(j)})^2 )
    
    where L_b = -mean_i(log p_g(x_i, y_i)) for batch b.
    """
    fisher_diag = {}

    for name, param in model.flow.named_parameters():
        fisher_diag['flow.' + name] = torch.zeros_like(param.data)

    model.classifier.eval()
    model.flow.eval()

    n_batches = 0
    for batch_idx, (x, y) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        # Skip degenerate batches (fewer than 2 samples)
        if x.shape[0] < 2:
            continue

        with torch.no_grad():
            xa = model.classifier.forward_to_xa(x)
            xa = xa.reshape(xa.shape[0], -1)

        y_one_hot = F.one_hot(y, num_classes=num_classes).float()

        model.flow.zero_grad()
        log_prob = model.flow.log_prob(inputs=xa.detach(), context=y_one_hot)
        loss = -log_prob.mean()
        loss.backward()

        for name, param in model.flow.named_parameters():
            if param.grad is not None:
                fisher_diag['flow.' + name] += param.grad.data ** 2

        n_batches += 1

    if n_batches > 0:
        for name in fisher_diag:
            fisher_diag[name] /= n_batches
    else:
        # No usable batches — Fisher is uninformative, use uniform weights
        import glog as logger
        logger.warning('Fisher estimation got 0 usable batches; using uniform weights')

    # Add small constant for numerical stability (prevents division by zero)
    for name in fisher_diag:
        fisher_diag[name] += 1e-8

    return fisher_diag


def fisher_weighted_aggregate(server_model, selected_users, device, num_classes):
    """
    Aggregate NF parameters using Fisher-weighted averaging:
    
        θ_server = (Σ_k F_k)^{-1} * Σ_k (F_k * θ_k)
    
    Non-flow parameters (classifier) are still averaged with standard FedAvg.
    
    Args:
        server_model:   Server's PreciseModel
        selected_users: List of UserPreciseFCL objects
        device:         torch device
        num_classes:    Total number of classes
    """
    # ---- Standard FedAvg for classifier parameters ----
    classifier_param_dict = {}
    for name, param in server_model.classifier.named_parameters():
        classifier_param_dict['classifier.' + name] = torch.zeros_like(param.data)

    total_train = sum(u.train_samples for u in selected_users)
    for user in selected_users:
        ratio = user.train_samples / total_train
        for name, param in user.model.classifier.named_parameters():
            classifier_param_dict['classifier.' + name] += param.data * ratio

    for name, param in server_model.classifier.named_parameters():
        param.data = classifier_param_dict['classifier.' + name]

    # ---- Fisher-weighted aggregation for flow parameters ----
    if server_model.flow is None:
        return

    # Step 1: Compute Fisher diagonal for each client
    fisher_diags = []
    for user in selected_users:
        fisher = compute_diagonal_fisher_efficient(
            user.model, user.trainloader, device, num_classes, max_batches=5)
        fisher_diags.append(fisher)

    # Step 2: Compute weighted sum and Fisher sum
    flow_param_weighted = {}
    flow_fisher_sum = {}

    for name, param in server_model.flow.named_parameters():
        full_name = 'flow.' + name
        flow_param_weighted[full_name] = torch.zeros_like(param.data)
        flow_fisher_sum[full_name] = torch.zeros_like(param.data)

    for user_idx, user in enumerate(selected_users):
        fisher = fisher_diags[user_idx]
        for name, param in user.model.flow.named_parameters():
            full_name = 'flow.' + name
            if full_name in fisher:
                flow_param_weighted[full_name] += fisher[full_name] * param.data
                flow_fisher_sum[full_name] += fisher[full_name]

    # Step 3: Divide to get Fisher-weighted average
    for name, param in server_model.flow.named_parameters():
        full_name = 'flow.' + name
        if full_name in flow_fisher_sum:
            param.data = flow_param_weighted[full_name] / flow_fisher_sum[full_name]
