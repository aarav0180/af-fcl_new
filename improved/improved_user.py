"""
ImprovedUserPreciseFCL: Drop-in replacement for UserPreciseFCL
==============================================================

Extends UserPreciseFCL to route through ImprovedPreciseModel's methods.
Handles:
  - Passing server_flow to model.train_a_batch (Feature 2)
  - Passing last_flow to classifier training (Feature 5)
  - EMA initialization on task transitions (Feature 3)

If no features are enabled, behaves identically to the original.
"""

import copy
import torch
import glog as logger
import numpy as np

from FLAlgorithms.users.userPreciseFCL import UserPreciseFCL
from utils.meter import Meter

eps = 1e-30


class ImprovedUserPreciseFCL(UserPreciseFCL):
    """
    Extended user class for AF-FCL improvements.
    
    Key changes from original UserPreciseFCL:
      1. train() passes server_flow and last_flow through to model
      2. next_task() re-initializes EMA extractor (Feature 3)
      3. Stores reference to server flow for KL regularization (Feature 2)
    """

    def __init__(self, args, id, model, train_data, test_data, label_info,
                 use_adam=False, my_model_name=None, unique_labels=None,
                 classifier_head_list=[]):
        super().__init__(
            args, id, model, train_data, test_data, label_info,
            use_adam=use_adam, my_model_name=my_model_name,
            unique_labels=unique_labels,
            classifier_head_list=classifier_head_list)

        # Feature flags
        self.use_ema_extractor = getattr(args, 'use_ema_extractor', False)
        self.use_personalized_nf = getattr(args, 'use_personalized_nf', False)
        self.use_adaptive_theta = getattr(args, 'use_adaptive_theta', False)

        # Server flow reference for Feature 2 (set externally by server)
        self.server_flow = None

    def next_task(self, train, test, label_info=None, if_label=True):
        """
        Override: reinitialize EMA extractor at task boundary (Feature 3).
        """
        # Call original next_task
        super().next_task(train, test, label_info=label_info, if_label=if_label)

        # Feature 3: Re-init EMA with current classifier at each new task
        if self.use_ema_extractor:
            self.model.init_ema_extractor()
            if self.model.ema_extractor is not None:
                device = next(self.model.classifier.parameters()).device
                self.model.ema_extractor.to(device)

    def train(self, glob_iter, glob_iter_task, global_classifier, verbose):
        """
        Extended training loop.
        
        Changes from original:
          - Passes server_flow to model.train_a_batch (Feature 2)
          - Passes last_flow to classifier training (Feature 5)
          - Initializes EMA on first call (Feature 3)
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Feature 3: Initialize EMA if not yet done (first task)
        if self.use_ema_extractor and self.model.ema_extractor is None:
            self.model.init_ema_extractor()
            if self.model.ema_extractor is not None:
                self.model.ema_extractor.to(device)

        correct = 0
        sample_num = 0
        cls_meter = Meter()

        for iteration in range(self.local_epochs):
            samples = self.get_next_train_batch(count_labels=True)
            x, y = samples['X'].to(device), samples['y'].to(device)

            last_classifier = None
            last_flow = None
            if type(self.last_copy) != type(None):
                last_classifier = self.last_copy.classifier
                last_classifier.eval()
                if self.algorithm == 'PreciseFCL':
                    last_flow = self.last_copy.flow
                    last_flow.eval()

            # ---- Flow training phase ----
            if self.algorithm == 'PreciseFCL' and self.k_loss_flow > 0:
                self.model.classifier.eval()
                self.model.flow.train()

                # Route through improved model (handles F2, F3 internally)
                flow_result = self.model.train_a_batch(
                    x, y, train_flow=True, flow=None, last_flow=last_flow,
                    last_classifier=last_classifier,
                    global_classifier=global_classifier,
                    classes_so_far=self.classes_so_far,
                    classes_past_task=self.classes_past_task,
                    available_labels=self.available_labels,
                    available_labels_past=self.available_labels_past,
                    server_flow=self.server_flow)  # Feature 2
                cls_meter._update(flow_result, batch_size=x.shape[0])

            # ---- Classifier training phase ----
            flow = None
            if self.algorithm == 'PreciseFCL':
                if self.use_lastflow_x:
                    flow = last_flow
                else:
                    flow = self.model.flow
                    flow.eval()

            self.model.classifier.train()

            # Route through improved model (handles F1, F5, F6 internally)
            # Note: last_flow is passed as extra arg for adaptive theta (F5)
            cls_result = self.model.train_a_batch(
                x, y, train_flow=False, flow=flow, last_flow=last_flow,
                last_classifier=last_classifier,
                global_classifier=global_classifier,
                classes_so_far=self.classes_so_far,
                classes_past_task=self.classes_past_task,
                available_labels=self.available_labels,
                available_labels_past=self.available_labels_past,
                server_flow=self.server_flow)

            correct += cls_result['correct']
            sample_num += x.shape[0]
            cls_meter._update(cls_result, batch_size=x.shape[0])

        acc = float(correct) / sample_num
        result_dict = cls_meter.get_scalar_dict('global_avg')
        if 'flow_loss' not in result_dict.keys():
            result_dict['flow_loss'] = 0
        if 'flow_loss_last' not in result_dict.keys():
            result_dict['flow_loss_last'] = 0

        if verbose:
            logger.info((
                "Training for user {:d}; Acc: {:.2f} %%; c_loss: {:.4f}; "
                "kd_loss: {:.4f}; flow_prob_mean: {:.4f}; "
                "flow_loss: {:.4f}; flow_loss_last: {:.4f}; "
                "c_loss_flow: {:.4f}; kd_loss_flow: {:.4f}; "
                "kd_loss_feature: {:.4f}; kd_loss_output: {:.4f}").format(
                self.id, acc * 100.0,
                result_dict['c_loss'], result_dict['kd_loss'],
                result_dict['flow_prob_mean'],
                result_dict['flow_loss'], result_dict['flow_loss_last'],
                result_dict['c_loss_flow'], result_dict['kd_loss_flow'],
                result_dict['kd_loss_feature'], result_dict['kd_loss_output']))

        return {
            'acc': acc, 'c_loss': result_dict['c_loss'],
            'kd_loss': result_dict['kd_loss'],
            'flow_prob_mean': result_dict['flow_prob_mean'],
            'flow_loss': result_dict['flow_loss'],
            'flow_loss_last': result_dict['flow_loss_last'],
            'c_loss_flow': result_dict['c_loss_flow'],
            'kd_loss_flow': result_dict['kd_loss_flow'],
        }
