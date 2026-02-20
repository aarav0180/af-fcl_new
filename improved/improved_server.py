"""
ImprovedFedPrecise: Drop-in replacement for FedPrecise server
=============================================================

Extends FedPrecise and conditionally activates:
  F2 (--use_personalized_nf):   Sends server flow to clients for KL reg
  F4 (--use_fisher_aggregation): Fisher-weighted NF aggregation

If no features are enabled, behaves identically to the original.
"""

import copy
import glog as logger
import torch
import time
import numpy as np

from FLAlgorithms.servers.serverPreciseFCL import FedPrecise
from improved.improved_model import ImprovedPreciseModel
from improved.improved_user import ImprovedUserPreciseFCL
from improved.features.fisher_aggregation import fisher_weighted_aggregate
from utils.dataset import get_dataset
from utils.model_utils import read_user_data_PreciseFCL
from utils.utils import str_in_list


class ImprovedFedPrecise(FedPrecise):
    """
    Extended server with modular improvements.
    
    Feature flags:
      args.use_personalized_nf     → Feature 2: send server flow to clients
      args.use_fisher_aggregation  → Feature 4: Fisher-weighted NF aggregation
    """

    def __init__(self, args, model: ImprovedPreciseModel, seed):
        # We override init_users, so call grandparent (Server) init manually
        # then replicate the FedPrecise-specific init
        from FLAlgorithms.servers.serverbase import Server
        Server.__init__(self, args, model, seed)

        self.classifier_global_mode = args.classifier_global_mode
        self.use_adam = 'adam' in self.algorithm.lower()
        self.data = get_dataset(args, args.dataset, args.datadir, args.data_split_file)
        self.unique_labels = self.data['unique_labels']
        self.classifier_head_list = ['classifier.fc_classifier', 'classifier.fc2']

        # Feature flags
        self.use_fisher_aggregation = getattr(args, 'use_fisher_aggregation', False)
        self.use_personalized_nf = getattr(args, 'use_personalized_nf', False)

        # Initialize IMPROVED users
        self._init_improved_users(self.data, args, model)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info('Using device: ' + str(device))
        self.device = device

        for u in self.users:
            u.model = u.model.to(device)

        self.model.to(device)

    def _init_improved_users(self, data, args, model):
        """Initialize ImprovedUserPreciseFCL instead of UserPreciseFCL."""
        self.users = []
        total_users = len(data['client_names'])
        for i in range(total_users):
            id, train_data, test_data, label_info = read_user_data_PreciseFCL(
                i, data, dataset=args.dataset, count_labels=True, task=0)

            self.total_train_samples += len(train_data)
            self.total_test_samples += len(test_data)
            id = i

            user = ImprovedUserPreciseFCL(
                args,
                id,
                model,
                train_data,
                test_data,
                label_info,
                use_adam=self.use_adam,
                my_model_name='fedprecise',
                unique_labels=self.unique_labels,
                classifier_head_list=self.classifier_head_list,
            )

            self.users.append(user)
            user.classes_so_far.extend(label_info['labels'])
            user.current_labels.extend(label_info['labels'])

        logger.info("Number of Train/Test samples: %d/%d" % (
            self.total_train_samples, self.total_test_samples))
        logger.info("Data from {} users in total.".format(total_users))
        logger.info("Finished creating ImprovedFedPrecise server.")

    def train(self, args):
        """
        Extended training loop:
          - Feature 2: Passes server flow copy to users before training
          - Feature 4: Uses Fisher-weighted aggregation for NF params
        """
        N_TASKS = len(
            self.data['train_data'][self.data['client_names'][0]]['x'])

        for task in range(N_TASKS):

            # ===================
            # The First Task
            # ===================
            if task == 0:
                available_labels = set()
                available_labels_current = set()
                available_labels_past = set()
                for u in self.users:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(
                        set(u.current_labels))

                for u in self.users:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

            # ===================
            # Initialize new Task
            # ===================
            else:
                self.current_task = task
                torch.cuda.empty_cache()

                for i in range(len(self.users)):
                    id, train_data, test_data, label_info = read_user_data_PreciseFCL(
                        i, self.data, dataset=args.dataset,
                        count_labels=True, task=task)
                    self.users[i].next_task(train_data, test_data, label_info)

                available_labels = set()
                available_labels_current = set()
                available_labels_past = self.users[0].available_labels
                for u in self.users:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(
                        set(u.current_labels))

                for u in self.users:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

            # ===================
            #    print info.
            # ===================
            if True:
                for u in self.users:
                    logger.info("classes so far: " + str(u.classes_so_far))
                logger.info("available labels for the Client: " + str(
                    self.users[-1].available_labels))
                logger.info("available labels (current) for the Client: " + str(
                    self.users[-1].available_labels_current))

            # ============ train ==============
            epoch_per_task = int(self.num_glob_iters / N_TASKS)

            for glob_iter_task in range(epoch_per_task):
                glob_iter = glob_iter_task + (epoch_per_task) * task

                logger.info(
                    "\n\n------------- Round number: %d | Current task: %d "
                    "-------------\n\n" % (glob_iter, task))

                self.selected_users, self.user_idxs = self.select_users(
                    glob_iter, len(self.users), return_idx=True)

                if self.algorithm != 'local':
                    self.send_parameters(mode='all', beta=1)

                # ---- Feature 2: Send server flow to clients for KL ----
                if self.use_personalized_nf and self.model.flow is not None:
                    server_flow_copy = copy.deepcopy(self.model.flow)
                    server_flow_copy.eval()
                    for p in server_flow_copy.parameters():
                        p.requires_grad_(False)
                    for user in self.selected_users:
                        user.server_flow = server_flow_copy
                else:
                    for user in self.selected_users:
                        user.server_flow = None

                chosen_verbose_user = np.random.randint(0, len(self.users))
                self.timestamp = time.time()

                # ---- Train users ----
                self.pickle_record['train'][glob_iter] = {}

                global_classifier = self.model.classifier
                global_classifier.eval()

                for user_id, user in zip(self.user_idxs, self.selected_users):
                    verbose = True

                    user_result = user.train(
                        glob_iter,
                        glob_iter_task,
                        global_classifier,
                        verbose=verbose)

                    self.pickle_record['train'][glob_iter][user_id] = user_result

                curr_timestamp = time.time()
                train_time = (curr_timestamp - self.timestamp) / len(
                    self.selected_users)
                self.metrics['user_train_time'].append(train_time)

                self.timestamp = time.time()

                # ==================
                # Server aggregation
                # ==================
                if self.algorithm != 'local':
                    # Feature 4: Fisher-weighted aggregation
                    if self.use_fisher_aggregation:
                        fisher_weighted_aggregate(
                            self.model, self.selected_users,
                            self.device, self.unique_labels)
                    else:
                        # Original aggregation
                        self.aggregate_parameters_(class_partial=False)

                curr_timestamp = time.time()
                agg_time = curr_timestamp - self.timestamp
                self.metrics['server_agg_time'].append(agg_time)

            if self.algorithm != 'local':
                self.send_parameters(mode='all', beta=1)

            self.evaluate_all_(glob_iter=glob_iter, matrix=True, personal=False)
            self.save_pickle()
