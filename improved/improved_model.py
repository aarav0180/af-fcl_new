"""
ImprovedPreciseModel: Drop-in replacement for PreciseModel
==========================================================

Extends PreciseModel and conditionally activates improvements based on 
feature flags in args. If no flags are enabled, behaves identically to 
the original PreciseModel.

Features integrated:
  F1 (--use_density_ratio):  density_ratio_credibility replaces probability_in_localdata
  F2 (--use_personalized_nf): train_flow_with_kl replaces train_a_batch_flow
  F3 (--use_ema_extractor):  EMA-based flow training & KD
  F5 (--use_adaptive_theta): dynamic theta replaces fixed flow_explore_theta
  F6 (--use_sinkhorn_kd):   sinkhorn_feature_distillation replaces MSE KD
  
  (F4 is server-side only, handled in improved_server.py)
"""

import torch
import numpy as np
import torch.nn.functional as F

from FLAlgorithms.PreciseFCLNet.model import PreciseModel, MultiClassCrossEntropy, eps
from utils.utils import myitem

# Feature modules
from improved.features.density_ratio import (
    sample_from_flow_with_noise,
    density_ratio_credibility,
)
from improved.features.personalized_nf import train_flow_with_kl
from improved.features.ema_extractor import (
    EMAFeatureExtractor,
    train_flow_with_ema,
    compute_kd_loss_ema,
)
from improved.features.adaptive_theta import (
    compute_adaptive_theta,
    compute_explore_forget_weight,
)
from improved.features.sinkhorn_kd import sinkhorn_feature_distillation


class ImprovedPreciseModel(PreciseModel):
    """
    Extended PreciseModel with modular research improvements.
    
    Feature flags are read from args:
      args.use_density_ratio      → Feature 1
      args.use_personalized_nf    → Feature 2  
      args.use_ema_extractor      → Feature 3
      args.use_adaptive_theta     → Feature 5
      args.use_sinkhorn_kd        → Feature 6
      
    If a flag is False, the corresponding original method is used.
    """

    def __init__(self, args):
        super().__init__(args)

        # Store feature flags
        self.use_density_ratio = getattr(args, 'use_density_ratio', False)
        self.use_personalized_nf = getattr(args, 'use_personalized_nf', False)
        self.use_ema_extractor = getattr(args, 'use_ema_extractor', False)
        self.use_adaptive_theta = getattr(args, 'use_adaptive_theta', False)
        self.use_sinkhorn_kd = getattr(args, 'use_sinkhorn_kd', False)

        # Feature-specific hyperparameters
        self.nf_kl_lambda = getattr(args, 'nf_kl_lambda', 0.1)
        self.ema_alpha = getattr(args, 'ema_alpha', 0.999)
        self.adaptive_theta_beta = getattr(args, 'adaptive_theta_beta', 1.0)
        self.sinkhorn_reg = getattr(args, 'sinkhorn_reg', 0.1)
        self.sinkhorn_iters = getattr(args, 'sinkhorn_iters', 30)

        # EMA extractor (Feature 3) — initialized later when moved to device
        self.ema_extractor = None

    def init_ema_extractor(self):
        """Initialize EMA extractor after model is on the correct device."""
        if self.use_ema_extractor and self.ema_extractor is None:
            self.ema_extractor = EMAFeatureExtractor(
                self.classifier, alpha=self.ema_alpha)

    def to(self, device):
        """Override to also move EMA extractor."""
        result = super().to(device)
        if self.ema_extractor is not None:
            self.ema_extractor.to(device)
        return result

    # =====================================================================
    # train_a_batch: Entry point — routes to classifier or flow training
    # =====================================================================

    def train_a_batch(self, x, y, train_flow, flow, last_flow,
                      last_classifier, global_classifier,
                      classes_so_far, classes_past_task,
                      available_labels, available_labels_past,
                      server_flow=None):
        """
        Extended train_a_batch with optional server_flow arg (Feature 2).
        Extra kwarg server_flow is ignored by original code paths.
        """
        if not train_flow:
            return self.train_a_batch_classifier(
                x, y, flow, last_flow, last_classifier, global_classifier,
                classes_past_task, available_labels)
        else:
            return self.train_a_batch_flow_improved(
                x, y, last_flow, classes_so_far, available_labels_past,
                server_flow=server_flow)

    # =====================================================================
    # Flow training (Features 2, 3)
    # =====================================================================

    def train_a_batch_flow_improved(self, x, y, last_flow, classes_so_far,
                                    available_labels_past, server_flow=None):
        """
        Improved flow training with optional:
          - EMA extractor (Feature 3): train NF on stable EMA features
          - Personalized NF + KL (Feature 2): add KL to server flow
        Falls back to original if neither enabled.
        """
        # Feature 3: Use EMA extractor for stable flow targets
        if self.use_ema_extractor and self.ema_extractor is not None:
            result = train_flow_with_ema(
                self, self.ema_extractor, x, y, last_flow,
                classes_so_far, available_labels_past)
        # Feature 2: Add KL regularization to server flow
        elif self.use_personalized_nf and server_flow is not None:
            result = train_flow_with_kl(
                self, x, y, last_flow, server_flow,
                classes_so_far, available_labels_past,
                kl_lambda=self.nf_kl_lambda)
        else:
            # Original flow training
            result = self.train_a_batch_flow(
                x, y, last_flow, classes_so_far, available_labels_past)

        return result

    # =====================================================================
    # Classifier training (Features 1, 5, 6)
    # =====================================================================

    def train_a_batch_classifier(self, x, y, flow, last_flow, last_classifier,
                                 global_classifier, classes_past_task,
                                 available_labels):
        """
        Improved classifier training with optional:
          - Density ratio credibility (Feature 1)
          - Adaptive theta (Feature 5)
          - Sinkhorn KD (Feature 6)
          
        Note: This has an extra arg `last_flow` compared to original to 
        support Feature 5 (adaptive theta needs the old flow).
        """

        if self.algorithm == 'PreciseFCL' and type(flow) != type(None) and self.k_loss_flow > 0:
            batch_size = x.shape[0]

            with torch.no_grad():
                _, xa, _ = self.classifier(x)
                xa = xa.reshape(xa.shape[0], -1)
                y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
                log_prob, xa_u = flow.log_prob_and_noise(xa, y_one_hot)
                log_prob = log_prob.detach()
                xa_u = xa_u.detach()
                prob_mean = torch.exp(log_prob / xa.shape[1]).mean() + eps

                # ---- Feature 1: Density Ratio Credibility ----
                if self.use_density_ratio:
                    # Sample with noise vectors for latent-space density ratio
                    flow_xa, label, _, flow_noise = sample_from_flow_with_noise(
                        flow, available_labels, batch_size, self.num_classes)
                    flow_xa_prob = density_ratio_credibility(
                        xa_u, y, prob_mean, flow_noise, label)
                else:
                    # Original: absolute density in mixed spaces
                    flow_xa, label, _ = self.sample_from_flow(
                        flow, available_labels, batch_size)
                    flow_xa_prob = self.probability_in_localdata(
                        xa_u, y, prob_mean, flow_xa, label)

                flow_xa_prob = flow_xa_prob.detach()
                flow_xa_prob_mean = flow_xa_prob.mean()

            flow_xa = flow_xa.reshape(flow_xa.shape[0], *self.xa_shape)
            softmax_output_flow, _ = self.classifier.forward_from_xa(flow_xa)
            c_loss_flow_generate = (
                self.classify_criterion_noreduce(
                    torch.log(softmax_output_flow + eps),
                    torch.Tensor(label).long().cuda()
                ) * flow_xa_prob
            ).mean()

            # ---- Feature 5: Adaptive Theta ----
            if self.use_adaptive_theta:
                adaptive_theta = compute_adaptive_theta(
                    self.classifier, last_flow, x, y,
                    self.num_classes, beta=self.adaptive_theta_beta)
                k_loss_flow_explore_forget = compute_explore_forget_weight(
                    adaptive_theta, prob_mean)
            else:
                # Original: fixed theta
                k_loss_flow_explore_forget = (
                    (1 - self.flow_explore_theta) * prob_mean + self.flow_explore_theta)

            kd_loss_output_last_flow, kd_loss_output_global_flow = \
                self.knowledge_distillation_on_output(
                    flow_xa, softmax_output_flow, last_classifier, global_classifier)
            kd_loss_flow = (kd_loss_output_last_flow + kd_loss_output_global_flow) * self.k_kd_output

            c_loss_flow = (
                c_loss_flow_generate * k_loss_flow_explore_forget + kd_loss_flow
            ) * self.k_loss_flow

            self.classifier_fb_optimizer.zero_grad()
            c_loss_flow.backward()
            self.classifier_fb_optimizer.step()
        else:
            prob_mean = 0.0
            c_loss_flow = 0.0
            kd_loss_flow = 0.0
            flow_xa_prob_mean = 0.0

        # ---- Standard classification loss ----
        softmax_output, xa, logits = self.classifier(x)
        c_loss_cls = self.classify_criterion(torch.log(softmax_output + eps), y)

        if self.algorithm == 'PreciseFCL':
            # ---- Feature 6: Sinkhorn KD replaces MSE feature distillation ----
            if self.use_sinkhorn_kd:
                kd_loss_feature_last, kd_loss_output_last, \
                    kd_loss_feature_global, kd_loss_output_global = \
                    self._kd_with_sinkhorn(x, xa, softmax_output,
                                           last_classifier, global_classifier)
            # ---- Feature 3: EMA KD replaces h'_a distillation ----
            elif self.use_ema_extractor and self.ema_extractor is not None:
                kd_loss_feature_last = compute_kd_loss_ema(
                    xa, self.ema_extractor, x, self.k_kd_last_cls)
                # Output KD still uses original method
                if self.k_kd_last_cls > 0 and type(last_classifier) != type(None):
                    softmax_output_last, _, _ = last_classifier(x)
                    softmax_output_last = softmax_output_last.detach()
                    kd_loss_output_last = self.k_kd_last_cls * MultiClassCrossEntropy(
                        softmax_output, softmax_output_last, T=2)
                else:
                    kd_loss_output_last = 0

                if self.k_kd_global_cls > 0:
                    softmax_output_global, _, _ = global_classifier(x)
                    softmax_output_global = softmax_output_global.detach()
                    kd_loss_feature_global = 0  # EMA handles feature KD
                    kd_loss_output_global = self.k_kd_global_cls * MultiClassCrossEntropy(
                        softmax_output, softmax_output_global, T=2)
                else:
                    kd_loss_feature_global = 0
                    kd_loss_output_global = 0
            else:
                # Original KD
                kd_loss_feature_last, kd_loss_output_last, \
                    kd_loss_feature_global, kd_loss_output_global = \
                    self.knowledge_distillation_on_xa_output(
                        x, xa, softmax_output, last_classifier, global_classifier)

            kd_loss_feature = (kd_loss_feature_last + kd_loss_feature_global) * self.k_kd_feature
            kd_loss_output = (kd_loss_output_last + kd_loss_output_global) * self.k_kd_output
            kd_loss = kd_loss_feature + kd_loss_output
        else:
            kd_loss_feature, kd_loss_output, kd_loss = 0, 0, 0

        c_loss = c_loss_cls + kd_loss
        correct = (torch.sum(torch.argmax(softmax_output, dim=1) == y)).item()

        self.classifier_optimizer.zero_grad()
        c_loss.backward()
        self.classifier_optimizer.step()

        # ---- Feature 3: Update EMA after classifier step ----
        if self.use_ema_extractor and self.ema_extractor is not None:
            self.ema_extractor.update(self.classifier)

        prob_mean = myitem(prob_mean)
        c_loss_flow = myitem(c_loss_flow)
        kd_loss = myitem(kd_loss)
        kd_loss_flow = myitem(kd_loss_flow)
        kd_loss_feature = myitem(kd_loss_feature)
        kd_loss_output = myitem(kd_loss_output)

        return {
            'c_loss': c_loss.item(), 'kd_loss': kd_loss, 'correct': correct,
            'flow_prob_mean': flow_xa_prob_mean,
            'c_loss_flow': c_loss_flow, 'kd_loss_flow': kd_loss_flow,
            'kd_loss_feature': kd_loss_feature, 'kd_loss_output': kd_loss_output,
        }

    def _kd_with_sinkhorn(self, x, xa, softmax_output,
                          last_classifier, global_classifier):
        """
        Knowledge distillation using Sinkhorn divergence for feature loss
        and original MultiClassCrossEntropy for output loss.
        """
        # Last classifier KD
        if self.k_kd_last_cls > 0 and type(last_classifier) != type(None):
            softmax_output_last, xa_last, _ = last_classifier(x)
            xa_last = xa_last.detach()
            softmax_output_last = softmax_output_last.detach()
            # Feature 6: Sinkhorn replaces MSE for feature KD
            kd_loss_feature_last = sinkhorn_feature_distillation(
                xa, xa_last, self.k_kd_last_cls,
                reg=self.sinkhorn_reg, max_iter=self.sinkhorn_iters)
            kd_loss_output_last = self.k_kd_last_cls * MultiClassCrossEntropy(
                softmax_output, softmax_output_last, T=2)
        else:
            kd_loss_feature_last = 0
            kd_loss_output_last = 0

        # Global classifier KD
        if self.k_kd_global_cls > 0:
            softmax_output_global, xa_global, _ = global_classifier(x)
            xa_global = xa_global.detach()
            softmax_output_global = softmax_output_global.detach()
            kd_loss_feature_global = sinkhorn_feature_distillation(
                xa, xa_global, self.k_kd_global_cls,
                reg=self.sinkhorn_reg, max_iter=self.sinkhorn_iters)
            kd_loss_output_global = self.k_kd_global_cls * MultiClassCrossEntropy(
                softmax_output, softmax_output_global, T=2)
        else:
            kd_loss_feature_global = 0
            kd_loss_output_global = 0

        return (kd_loss_feature_last, kd_loss_output_last,
                kd_loss_feature_global, kd_loss_output_global)
