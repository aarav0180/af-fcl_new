"""
Feature 3: EMA (Exponential Moving Average) Feature Extractor
=============================================================

The NF is trained to model p(h_a(x), y) — but h_a is being updated
simultaneously. This means the NF chases a MOVING TARGET: every update
to h_a changes what the NF should be modeling.

The KD loss (Eq. 6) partially restrains h_a but creates a deadlock:
  - Strong KD → h_a can't adapt → poor new-task performance
  - Weak KD → h_a drifts → NF becomes invalid

Fix: Maintain an EMA of the feature extractor h_a as a stable target:
    h_a^{EMA}(τ) = α · h_a^{EMA}(τ-1) + (1-α) · h_a(τ)

Train the NF on features from h_a^{EMA} (temporally smooth, stable).
The classifier's h_a is free to adapt rapidly.

This is a two-timescale dynamic (cf. Mean Teacher, Tarvainen & Valpola, 2017).
The EMA model is guaranteed to be more stable than any single iterate,
providing a Lyapunov-stable target for the NF.
"""

import copy
import torch


class EMAFeatureExtractor:
    """
    Maintains an Exponential Moving Average of the classifier's feature
    extractor h_a, to be used as a stable training target for the NF.
    
    The EMA classifier is a full deep copy, but only its feature extractor
    weights (conv1, conv2, conv3, fc1 for CNN; features for ResNet) are
    tracked. The classification head (fc2, fc_classifier) is NOT EMA'd.
    """

    def __init__(self, classifier, alpha=0.999):
        """
        Args:
            classifier: The current classifier module (S_ConvNet or Resnet_plus)
            alpha:      EMA decay rate. Higher = more stable, slower to track.
                        Typical: 0.999 for long training, 0.99 for short.
        """
        self.alpha = alpha
        # Deep copy the entire classifier to serve as EMA model
        self.ema_classifier = copy.deepcopy(classifier)
        # Freeze all EMA parameters (no gradients)
        for p in self.ema_classifier.parameters():
            p.requires_grad_(False)
        self.ema_classifier.eval()

    def update(self, classifier):
        """
        Update EMA parameters: θ_ema = α * θ_ema + (1-α) * θ_current
        
        Call this AFTER each optimizer step on the classifier.
        """
        with torch.no_grad():
            for ema_param, current_param in zip(
                    self.ema_classifier.parameters(), classifier.parameters()):
                ema_param.data.mul_(self.alpha).add_(
                    current_param.data, alpha=1.0 - self.alpha)

    def forward_to_xa(self, x):
        """
        Compute features using EMA classifier's feature extractor.
        Returns features in the same space as classifier.forward_to_xa(x)
        but from the temporally stable EMA model.
        """
        self.ema_classifier.eval()
        with torch.no_grad():
            xa_ema = self.ema_classifier.forward_to_xa(x)
        return xa_ema

    def to(self, device):
        """Move EMA classifier to device."""
        self.ema_classifier = self.ema_classifier.to(device)
        return self

    def state_dict(self):
        """Return EMA classifier state dict for checkpointing."""
        return self.ema_classifier.state_dict()

    def load_state_dict(self, state_dict):
        """Load EMA classifier state dict."""
        self.ema_classifier.load_state_dict(state_dict)


def train_flow_with_ema(model, ema_extractor, x, y, last_flow,
                        classes_so_far, available_labels_past):
    """
    Train the NF using EMA feature extractor (stable target) instead of 
    the live classifier's h_a.
    
    Modified Eq. 5:
        L_NF^{EMA} = -1/|D^t_k| * sum log p_z(h_a^{EMA}(x_i), y_i) 
                     -1/|G_z|   * sum log p_z(z_i, y_i)
    
    Args:
        model:           PreciseModel instance
        ema_extractor:   EMAFeatureExtractor instance
        x, y:            Current batch
        last_flow:       NF from previous task
        classes_so_far:  All classes seen by this client
        available_labels_past: Labels from T-1
        
    Returns:
        dict with flow loss components
    """
    import torch.nn.functional as F
    from utils.utils import myitem

    # Use EMA extractor for stable features (no gradient to h_a)
    xa_ema = ema_extractor.forward_to_xa(x)
    xa_ema = xa_ema.reshape(xa_ema.shape[0], -1)

    y_one_hot = F.one_hot(y, num_classes=model.num_classes).float()

    # NF trained on stable EMA features
    loss_data = -model.flow.log_prob(inputs=xa_ema, context=y_one_hot).mean()

    # Replay from previous flow
    if model.algorithm == 'PreciseFCL' and type(last_flow) != type(None):
        batch_size = x.shape[0]
        with torch.no_grad():
            flow_xa, label, label_one_hot = model.sample_from_flow(
                last_flow, available_labels_past, batch_size)
        loss_last_flow = -model.flow.log_prob(inputs=flow_xa, context=label_one_hot).mean()
    else:
        loss_last_flow = 0

    loss_last_flow = model.k_flow_lastflow * loss_last_flow
    loss = loss_data + loss_last_flow

    model.flow_optimizer.zero_grad()
    loss.backward()
    model.flow_optimizer.step()

    return {'flow_loss': loss_data.item(), 'flow_loss_last': myitem(loss_last_flow)}


def compute_kd_loss_ema(xa_current, ema_extractor, x, k_kd_last_cls):
    """
    Compute KD loss comparing current h_a to EMA h_a (instead of h'_a).
    
    Modified Eq. 6:
        L_KD^{EMA} = ||h_a(x) - h_a^{EMA}(x)||^2
        
    This eliminates the deadlock: the EMA smoothly tracks the classifier,
    allowing h_a to adapt freely while the NF has a stable target.
    
    Args:
        xa_current:     Features from current classifier h_a(x) [N, D]
        ema_extractor:  EMAFeatureExtractor instance
        x:              Input data [N, C, H, W]
        k_kd_last_cls:  KD loss weight coefficient
        
    Returns:
        kd_loss_feature: Scalar feature distillation loss
    """
    xa_ema = ema_extractor.forward_to_xa(x)
    xa_ema = xa_ema.reshape(xa_ema.shape[0], -1).detach()
    
    kd_loss_feature = k_kd_last_cls * torch.pow(xa_ema - xa_current, 2).mean()
    return kd_loss_feature
