"""
Feature 2: Personalized NF with KL Regularization
==================================================

The paper uses one global NF aggregated via FedAvg for both:
  1. Generating replay features for all clients
  2. Credibility estimation (local Gaussian fit in latent space, Eq. 7-8)

These roles conflict: the global NF represents ALL clients' distributions,
but credibility estimation needs the NF to reflect the LOCAL client's 
feature geometry.

Fix: Each client trains a personalized NF g_k regularized to stay close
to the server NF g_server via KL divergence:

    L_{k,NF}^{pFL} = L_NF(g_k; D^t_k, G_z) + lambda * D_KL(g_k || g_server)

The KL between two normalizing flows is EXACTLY computable:
    D_KL(g_k || g_server) = E_{x~local}[ log p_{g_k}(x) - log p_{g_server}(x) ]

Both log-probs are tractable via the change-of-variables formula (Eq. 3).

- The personalized NF g_k is used for credibility estimation (accurate for local data)
- The global NF g_server (FedAvg of all g_k) is used for generative replay
- lambda scales with 1/n^t_k so data-poor clients stay closer to global
"""

import torch
import torch.nn.functional as F


def compute_nf_kl_divergence(local_flow, server_flow, features, labels, num_classes):
    """
    Compute KL(g_local || g_server) using local data features.
    
    D_KL = E_{z~local}[ log p_{g_local}(z) - log p_{g_server}(z) ]
    
    Both log-probabilities are exact (not approximate) for normalizing flows.
    
    Args:
        local_flow:   The client's personal NF model
        server_flow:  The server's global NF model (frozen, no grad)
        features:     Feature vectors h_a(x) from local data [N, D]
        labels:       Class labels [N]
        num_classes:  Total number of classes
        
    Returns:
        kl_div: Scalar KL divergence estimate
    """
    y_one_hot = F.one_hot(labels, num_classes=num_classes).float()

    # log p_{g_local}(z) — this gets gradients for local flow
    log_prob_local = local_flow.log_prob(inputs=features, context=y_one_hot)

    # log p_{g_server}(z) — no gradients, server flow is fixed
    with torch.no_grad():
        log_prob_server = server_flow.log_prob(inputs=features, context=y_one_hot)

    # D_KL = E[ log p_local - log p_server ]
    kl_div = (log_prob_local - log_prob_server).mean()

    return kl_div


def train_flow_with_kl(model, x, y, last_flow, server_flow,
                       classes_so_far, available_labels_past,
                       kl_lambda=0.1):
    """
    Train the NF with additional KL regularization toward server NF.
    
    Modified version of PreciseModel.train_a_batch_flow that adds:
        L_total = L_data + k_flow_lastflow * L_lastflow + kl_lambda * D_KL(g_k || g_server)
    
    Args:
        model:                The PreciseModel instance  
        x, y:                 Current batch data and labels
        last_flow:            NF from previous task (for replay)
        server_flow:          Server's global NF (for KL regularization)
        classes_so_far:       All classes seen by this client
        available_labels_past: Labels available from all clients on T-1
        kl_lambda:            Weight for KL term (default 0.1)
        
    Returns:
        dict with loss components
    """
    xa = model.classifier.forward_to_xa(x)
    xa = xa.reshape(xa.shape[0], -1)
    y_one_hot = F.one_hot(y, num_classes=model.num_classes).float()

    # Standard NF training on current data
    loss_data = -model.flow.log_prob(inputs=xa, context=y_one_hot).mean()

    # Replay loss from previous task's NF
    if model.algorithm == 'PreciseFCL' and type(last_flow) != type(None):
        batch_size = x.shape[0]
        with torch.no_grad():
            flow_xa, label, label_one_hot = model.sample_from_flow(
                last_flow, available_labels_past, batch_size)
        loss_last_flow = -model.flow.log_prob(inputs=flow_xa, context=label_one_hot).mean()
    else:
        loss_last_flow = 0

    loss_last_flow = model.k_flow_lastflow * loss_last_flow

    # KL regularization toward server NF
    if server_flow is not None and kl_lambda > 0:
        kl_loss = compute_nf_kl_divergence(
            model.flow, server_flow, xa.detach(), y, model.num_classes)
        loss_kl = kl_lambda * kl_loss
    else:
        loss_kl = torch.tensor(0.0, device=x.device)

    # Total loss
    loss = loss_data + loss_last_flow + loss_kl

    model.flow_optimizer.zero_grad()
    loss.backward()
    model.flow_optimizer.step()

    from utils.utils import myitem
    return {
        'flow_loss': loss_data.item(),
        'flow_loss_last': myitem(loss_last_flow),
        'flow_loss_kl': myitem(loss_kl),
    }
