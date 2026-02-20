#!/usr/bin/env python
import argparse
from utils.model_utils import create_model
import os
import glog as logger
import json

from FLAlgorithms.servers.serverPreciseFCL import FedPrecise
from utils.utils import setup_seed, set_log_file, print_args
from improved import any_improvement_enabled

def create_server_n_user(args, i):
    
    if any_improvement_enabled(args):
        # Use improved model + server when any feature flag is active
        from improved.improved_model import ImprovedPreciseModel
        from improved.improved_server import ImprovedFedPrecise
        model = ImprovedPreciseModel(args)
        server = ImprovedFedPrecise(args, model, i)
        enabled = [f for f in ['density_ratio', 'personalized_nf', 'ema_extractor',
                                'fisher_aggregation', 'adaptive_theta', 'sinkhorn_kd']
                   if getattr(args, f'use_{f}', False)]
        logger.info(f'[IMPROVED] Active features: {enabled}')
    else:
        # Original code path — no changes
        model = create_model(args)
        server = FedPrecise(args, model, i)
        logger.info('[ORIGINAL] Running original AF-FCL (no improvements enabled)')
    return server


def run_job(args, seed):

    logger.info('random seed is: %d'%(seed))
    logger.info("\n\n         [ Start training iteration, seed: {} ]           \n\n".format(seed))
    # Generate model
    server = create_server_n_user(args, seed)
    if args.train:
        server.train(args)

def main(args):
    run_job(args, args.seed)
    
    logger.info("Finished training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST-SVHN-FASHION", choices=['EMNIST-Letters', 'EMNIST-Letters-malicious', 
                                                                            'EMNIST-Letters-shuffle', 'CIFAR100', 'MNIST-SVHN-FASHION'])
    parser.add_argument("--datadir", type=str, default="datasets/PreciseFCL/")
    parser.add_argument("--data_split_file", type=str, default="data_split/MNISTSVHNFASHION_split_cn10_tn6_cet3_s2571.pkl")
    parser.add_argument("--malicious_client_num", type=int, default=0)
    parser.add_argument("--algorithm", type=str, default="PreciseFCL", choices=['PreciseFCL'])
    parser.add_argument("--seed", type=int, default=2571)

    # PreciseFCL
    parser.add_argument("--k_loss_flow", type=float, default=0.1)
    parser.add_argument("--k_kd_global_cls", type=float, default=0)
    parser.add_argument("--k_kd_last_cls", type=float, default=0.2)
    parser.add_argument("--k_kd_feature", type=float, default=0.5)
    parser.add_argument("--k_kd_output", type=float, default=0.1)
    parser.add_argument("--k_flow_lastflow", type=float, default=0.4)
    parser.add_argument("--flow_epoch", type=int, default=5)
    parser.add_argument("--flow_explore_theta", type=float, default=0.2)
    parser.add_argument("--classifier_global_mode", type=str, default='all', help='[head, extractor, none, all]')
    parser.add_argument('--flow_lr', type=float, default=1e-4)  
    parser.add_argument('--fedprox_k', type=float, default=0) 
    parser.add_argument('--use_lastflow_x', action="store_true") 

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-04)  
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight-decay', type=float, default=0)    

    parser.add_argument("--num_glob_iters", type=int, default=60)
    parser.add_argument("--local_epochs", type=int, default=100)

    parser.add_argument("--train", type=int, default=1, choices=[0,1])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"], help="run device (cpu | cuda)")
    
    # model
    parser.add_argument('--c-channel-size', type=int, default=64)
    parser.add_argument("--model", type=str, default="cnn")

    # run routine
    parser.add_argument('--target_dir_name', type = str, default="output_dir", help="the dim of the solution")
    parser.add_argument("--debug", action="store_true", help="debug or not")
    parser.add_argument("--ssh", action="store_true", help="whether is run by search")

    # ============================================================
    # Research Improvement Feature Flags (each is independently togglable)
    # ============================================================
    parser.add_argument('--use_all_improvements', action='store_true',
                        help='Enable ALL 6 research improvements at once')

    # Feature 1: Density ratio credibility (replaces Eq. 8 absolute density)
    parser.add_argument('--use_density_ratio', action='store_true',
                        help='F1: Use density ratio p_local/p_prior instead of absolute density')

    # Feature 2: Personalized NF with KL regularization to server
    parser.add_argument('--use_personalized_nf', action='store_true',
                        help='F2: Personalized NF per client with KL(g_k || g_server)')
    parser.add_argument('--nf_kl_lambda', type=float, default=0.1,
                        help='F2: Weight for KL divergence term in NF training')

    # Feature 3: EMA feature extractor for stable NF training
    parser.add_argument('--use_ema_extractor', action='store_true',
                        help='F3: Use EMA of h_a as stable NF training target')
    parser.add_argument('--ema_alpha', type=float, default=0.999,
                        help='F3: EMA decay rate (higher=more stable)')

    # Feature 4: Fisher-weighted aggregation for NF parameters
    parser.add_argument('--use_fisher_aggregation', action='store_true',
                        help='F4: Fisher Information weighted FedAvg for NF params')

    # Feature 5: Task-similarity adaptive explore-theta
    parser.add_argument('--use_adaptive_theta', action='store_true',
                        help='F5: Data-driven theta via cross-task NF log-likelihood')
    parser.add_argument('--adaptive_theta_beta', type=float, default=1.0,
                        help='F5: Sigmoid scaling for adaptive theta sensitivity')

    # Feature 6: Sinkhorn divergence for feature distillation (replaces Eq. 6)
    parser.add_argument('--use_sinkhorn_kd', action='store_true',
                        help='F6: Sinkhorn divergence replaces MSE in feature KD')
    parser.add_argument('--sinkhorn_reg', type=float, default=0.1,
                        help='F6: Entropic regularization epsilon for Sinkhorn')
    parser.add_argument('--sinkhorn_iters', type=int, default=30,
                        help='F6: Number of Sinkhorn iterations')

    args = parser.parse_args()

    # Convenience: --use_all_improvements enables all 6 features
    if args.use_all_improvements:
        args.use_density_ratio = True
        args.use_personalized_nf = True
        args.use_ema_extractor = True
        args.use_fisher_aggregation = True
        args.use_adaptive_theta = True
        args.use_sinkhorn_kd = True

    os.makedirs(args.target_dir_name, exist_ok=True)
    setup_seed(args.seed)
    # args.target_dir = '_'.join([args.target_dir, args.dataset, args.fed_alg])
    if not args.debug:
        log_name = 'run.log'
        args.log_pth = os.path.join(args.target_dir_name, log_name)
        set_log_file(args.log_pth, file_only=args.ssh)
    else:
        logger.info('------------------Debug--------------------')

    print_args(args)
    with open(os.path.join(args.target_dir_name, 'args.json'), "w") as f:
        json.dump(args.__dict__, f, indent =2)
        
    main(args)
