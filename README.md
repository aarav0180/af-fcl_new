# AF-FCL: Accurate Forgetting for Heterogeneous Federated Continual Learning

Implementation of AF-FCL (ICLR 2024) with six modular research improvements. Each improvement can be independently enabled via command-line flags. When no flags are set, the code runs the original AF-FCL baseline with no modifications.

---

## Requirements

```
pip install -r requirements.txt
```

Additional dependencies: `torch`, `torchvision`, `glog`. Install PyTorch following [pytorch.org](https://pytorch.org/get-started/locally/).

## Dataset Preparation

All datasets are automatically downloaded via `torchvision.datasets`. Data split files are located in `data_split/`.

---

## Research Improvements

Six independently togglable features, each enabled by a single CLI flag:

| Flag | Feature | Description |
|------|---------|-------------|
| `--use_density_ratio` | F1: Density Ratio Credibility | Replaces absolute density (Eq. 8) with density ratio p_local/p_prior for importance-corrected replay weighting |
| `--use_personalized_nf` | F2: Personalized NF + KL | Per-client normalizing flow regularized to server NF via exact KL divergence |
| `--use_ema_extractor` | F3: EMA Feature Extractor | Exponential moving average of h_a as stable NF training target, decoupling feature adaptation from NF training |
| `--use_fisher_aggregation` | F4: Fisher-Weighted FedAvg | Diagonal Fisher Information weighted aggregation for NF parameters instead of plain FedAvg |
| `--use_adaptive_theta` | F5: Adaptive Explore-Theta | Data-driven theta via cross-task NF log-likelihood, replacing the fixed hyperparameter |
| `--use_sinkhorn_kd` | F6: Sinkhorn Feature KD | Sinkhorn divergence replaces MSE in feature distillation (Eq. 6), allowing distributional rather than pointwise matching |

Use `--use_all_improvements` to enable all six features at once.

### Hyperparameters for Improvements

| Parameter | Default | Used By | Description |
|-----------|---------|---------|-------------|
| `--nf_kl_lambda` | 0.1 | F2 | Weight for KL divergence term in personalized NF training |
| `--ema_alpha` | 0.999 | F3 | EMA decay rate (higher is more stable) |
| `--adaptive_theta_beta` | 1.0 | F5 | Sigmoid scaling for adaptive theta sensitivity |
| `--sinkhorn_reg` | 0.1 | F6 | Entropic regularization epsilon for Sinkhorn iterations |
| `--sinkhorn_iters` | 30 | F6 | Number of Sinkhorn iterations |

---

## Quick Verification (CPU, No GPU)

Lightweight runs to verify the code executes correctly on a laptop. These use minimal epochs and iterations and are not intended for meaningful accuracy.

### Original baseline (CPU, debug mode)

```
python main.py --dataset MNIST-SVHN-FASHION --data_split_file data_split/MNISTSVHNFASHION_split_cn10_tn6_cet3_s2571.pkl --device cpu --num_glob_iters 2 --local_epochs 2 --batch_size 16 --debug
```

### Single improvement (CPU, debug mode)

```
python main.py --dataset MNIST-SVHN-FASHION --data_split_file data_split/MNISTSVHNFASHION_split_cn10_tn6_cet3_s2571.pkl --device cpu --num_glob_iters 2 --local_epochs 2 --batch_size 16 --debug --use_density_ratio
```

### All improvements (CPU, debug mode)

```
python main.py --dataset MNIST-SVHN-FASHION --data_split_file data_split/MNISTSVHNFASHION_split_cn10_tn6_cet3_s2571.pkl --device cpu --num_glob_iters 2 --local_epochs 2 --batch_size 16 --debug --use_all_improvements
```

### Test individual features one at a time (CPU, debug mode)

```
python main.py --dataset MNIST-SVHN-FASHION --data_split_file data_split/MNISTSVHNFASHION_split_cn10_tn6_cet3_s2571.pkl --device cpu --num_glob_iters 2 --local_epochs 2 --batch_size 16 --debug --use_personalized_nf --nf_kl_lambda 0.05
python main.py --dataset MNIST-SVHN-FASHION --data_split_file data_split/MNISTSVHNFASHION_split_cn10_tn6_cet3_s2571.pkl --device cpu --num_glob_iters 2 --local_epochs 2 --batch_size 16 --debug --use_ema_extractor --ema_alpha 0.99
python main.py --dataset MNIST-SVHN-FASHION --data_split_file data_split/MNISTSVHNFASHION_split_cn10_tn6_cet3_s2571.pkl --device cpu --num_glob_iters 2 --local_epochs 2 --batch_size 16 --debug --use_fisher_aggregation
python main.py --dataset MNIST-SVHN-FASHION --data_split_file data_split/MNISTSVHNFASHION_split_cn10_tn6_cet3_s2571.pkl --device cpu --num_glob_iters 2 --local_epochs 2 --batch_size 16 --debug --use_adaptive_theta --adaptive_theta_beta 1.0
python main.py --dataset MNIST-SVHN-FASHION --data_split_file data_split/MNISTSVHNFASHION_split_cn10_tn6_cet3_s2571.pkl --device cpu --num_glob_iters 2 --local_epochs 2 --batch_size 16 --debug --use_sinkhorn_kd --sinkhorn_reg 0.1 --sinkhorn_iters 20
```

---

## Full Training (GPU)

Standard training commands for reproducing results. Add `--use_all_improvements` or any combination of individual flags to enable improvements.

### EMNIST-Letters

Baseline:

```
python main.py --dataset EMNIST-Letters --data_split_file data_split/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl --device cuda --num_glob_iters 60 --local_epochs 100 --lr 1e-4 --flow_lr 1e-4 --k_loss_flow 0.5 --k_flow_lastflow 0.4 --flow_explore_theta 0 --target_dir_name output/emnist_baseline
```

With all improvements:

```
python main.py --dataset EMNIST-Letters --data_split_file data_split/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl --device cuda --num_glob_iters 60 --local_epochs 100 --lr 1e-4 --flow_lr 1e-4 --k_loss_flow 0.5 --k_flow_lastflow 0.4 --flow_explore_theta 0 --target_dir_name output/emnist_improved --use_all_improvements
```

### EMNIST-Shuffle

Baseline:

```
python main.py --dataset EMNIST-Letters-shuffle --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl --device cuda --num_glob_iters 60 --local_epochs 100 --lr 1e-4 --flow_lr 1e-3 --k_loss_flow 0.05 --k_flow_lastflow 0.02 --flow_explore_theta 0.5 --target_dir_name output/emnist_shuffle_baseline
```

With all improvements:

```
python main.py --dataset EMNIST-Letters-shuffle --data_split_file data_split/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl --device cuda --num_glob_iters 60 --local_epochs 100 --lr 1e-4 --flow_lr 1e-3 --k_loss_flow 0.05 --k_flow_lastflow 0.02 --flow_explore_theta 0.5 --target_dir_name output/emnist_shuffle_improved --use_all_improvements
```

### EMNIST-Noisy (M noisy clients)

Baseline:

```
python main.py --dataset EMNIST-Letters-malicious --data_split_file data_split/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl --device cuda --num_glob_iters 60 --local_epochs 100 --lr 1e-4 --flow_lr 1e-3 --k_loss_flow 0.5 --k_flow_lastflow 0.1 --flow_explore_theta 0.5 --malicious_client_num 2 --target_dir_name output/emnist_noisy_baseline
```

With all improvements:

```
python main.py --dataset EMNIST-Letters-malicious --data_split_file data_split/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl --device cuda --num_glob_iters 60 --local_epochs 100 --lr 1e-4 --flow_lr 1e-3 --k_loss_flow 0.5 --k_flow_lastflow 0.1 --flow_explore_theta 0.5 --malicious_client_num 2 --target_dir_name output/emnist_noisy_improved --use_all_improvements
```

### MNIST-SVHN-FASHION

Baseline:

```
python main.py --dataset MNIST-SVHN-FASHION --data_split_file data_split/MNISTSVHNFASHION_split_cn10_tn6_cet3_s2571.pkl --device cuda --num_glob_iters 60 --local_epochs 100 --lr 1e-4 --flow_lr 1e-3 --k_loss_flow 0.1 --k_flow_lastflow 0 --flow_explore_theta 0 --fedprox_k 0.001 --target_dir_name output/msf_baseline
```

With all improvements:

```
python main.py --dataset MNIST-SVHN-FASHION --data_split_file data_split/MNISTSVHNFASHION_split_cn10_tn6_cet3_s2571.pkl --device cuda --num_glob_iters 60 --local_epochs 100 --lr 1e-4 --flow_lr 1e-3 --k_loss_flow 0.1 --k_flow_lastflow 0 --flow_explore_theta 0 --fedprox_k 0.001 --target_dir_name output/msf_improved --use_all_improvements
```

### CIFAR100

Baseline:

```
python main.py --dataset CIFAR100 --data_split_file data_split/CIFAR100_split_cn10_tn4_cet20_s2571.pkl --device cuda --num_glob_iters 40 --local_epochs 400 --lr 1e-3 --flow_lr 5e-3 --k_loss_flow 0.5 --k_flow_lastflow 0.1 --flow_explore_theta 0.1 --fedprox_k 0.001 --target_dir_name output/cifar100_baseline
```

With all improvements:

```
python main.py --dataset CIFAR100 --data_split_file data_split/CIFAR100_split_cn10_tn4_cet20_s2571.pkl --device cuda --num_glob_iters 40 --local_epochs 400 --lr 1e-3 --flow_lr 5e-3 --k_loss_flow 0.5 --k_flow_lastflow 0.1 --flow_explore_theta 0.1 --fedprox_k 0.001 --target_dir_name output/cifar100_improved --use_all_improvements
```

---

## Combining Specific Features

Any subset of features can be combined. For example, density ratio with adaptive theta only:

```
python main.py --dataset MNIST-SVHN-FASHION --data_split_file data_split/MNISTSVHNFASHION_split_cn10_tn6_cet3_s2571.pkl --device cuda --num_glob_iters 60 --local_epochs 100 --lr 1e-4 --flow_lr 1e-3 --k_loss_flow 0.1 --k_flow_lastflow 0 --flow_explore_theta 0 --fedprox_k 0.001 --target_dir_name output/msf_f1_f5 --use_density_ratio --use_adaptive_theta
```

Fisher aggregation with Sinkhorn KD:

```
python main.py --dataset EMNIST-Letters --data_split_file data_split/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl --device cuda --num_glob_iters 60 --local_epochs 100 --lr 1e-4 --flow_lr 1e-4 --k_loss_flow 0.5 --k_flow_lastflow 0.4 --flow_explore_theta 0 --target_dir_name output/emnist_f4_f6 --use_fisher_aggregation --use_sinkhorn_kd --sinkhorn_reg 0.1
```

---

## Project Structure

```
AF-FCL/
    main.py                             Entry point with CLI argument parsing
    FLAlgorithms/
        PreciseFCLNet/
            model.py                    PreciseModel (classifier + normalizing flow)
            classify_net.py             S_ConvNet, Resnet_plus classifiers
        servers/
            serverbase.py               Server base class
            serverPreciseFCL.py         FedPrecise server (original)
        users/
            userbase.py                 User base class
            userPreciseFCL.py           UserPreciseFCL client (original)
    improved/
        __init__.py                     Feature flag registry
        improved_model.py               ImprovedPreciseModel (extends PreciseModel)
        improved_user.py                ImprovedUserPreciseFCL (extends UserPreciseFCL)
        improved_server.py              ImprovedFedPrecise (extends FedPrecise)
        features/
            density_ratio.py            F1: Density ratio credibility estimation
            personalized_nf.py          F2: Personalized NF with KL regularization
            ema_extractor.py            F3: EMA feature extractor
            fisher_aggregation.py       F4: Fisher-weighted NF aggregation
            adaptive_theta.py           F5: Task-similarity adaptive theta
            sinkhorn_kd.py              F6: Sinkhorn divergence feature distillation
    nflows/                             Normalizing flow library
    utils/                              Utilities, dataset loading, metrics
```

---

## Reference

The code structure is based on [FedCIL](https://github.com/daiqing98/FedCIL). The normalizing flow implementation refers to [nflows](https://github.com/bayesiains/nflows).