"""
Generate federated continual learning data-split files for AF-FCL.

Each .pkl file stores:
    train_inds   : list[client][task] -> list of int (indices into the
                   concatenated training dataset)
    test_inds    : list[client][task] -> list of int
    client_y_list: list[client][task] -> list of int (unique class labels
                   that client sees in that task)

Usage
-----
# Generate MNIST-SVHN-FASHION split (default)
python generate_splits.py

# Generate EMNIST-Letters split
python generate_splits.py --dataset EMNIST-Letters

# Generate CIFAR100 split
python generate_splits.py --dataset CIFAR100
"""

import argparse
import os
import pickle
import numpy as np
import torchvision.datasets as datasets
from torchvision import transforms


# ------------------------------------------------------------------ #
#  Dataset-specific configuration                                     #
# ------------------------------------------------------------------ #
CONFIGS = {
    "MNIST-SVHN-FASHION": dict(
        num_clients=10, num_tasks=6, classes_per_task=3,
        unique_labels=20, seed=2571,
        filename="MNISTSVHNFASHION_split_cn10_tn6_cet3_s2571.pkl",
    ),
    "EMNIST-Letters": dict(
        num_clients=8, num_tasks=6, classes_per_task=2,
        unique_labels=26, seed=2571,
        filename="EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl",
    ),
    "EMNIST-Letters-shuffle": dict(
        num_clients=8, num_tasks=6, classes_per_task=2,
        unique_labels=26, seed=2571,
        filename="EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl",
    ),
    "CIFAR100": dict(
        num_clients=10, num_tasks=4, classes_per_task=20,
        unique_labels=100, seed=2571,
        filename="CIFAR100_split_cn10_tn4_cet20_s2571.pkl",
    ),
}


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #
def load_labels(dataset_name, datadir):
    """Return (train_labels, test_labels) as numpy arrays for the
    *concatenated* dataset exactly as dataset.py builds it."""

    if dataset_name in ("EMNIST-Letters", "EMNIST-Letters-shuffle",
                         "EMNIST-Letters-malicious"):
        tr = datasets.EMNIST(datadir, "letters", download=True, train=True)
        te = datasets.EMNIST(datadir, "letters", download=True, train=False)
        # target_transform: x-1
        train_y = np.array([int(tr[i][1]) - 1 for i in range(len(tr))])
        test_y  = np.array([int(te[i][1]) - 1 for i in range(len(te))])

    elif dataset_name == "CIFAR100":
        tr = datasets.CIFAR100(datadir, download=True, train=True)
        te = datasets.CIFAR100(datadir, download=True, train=False)
        train_y = np.array([tr[i][1] for i in range(len(tr))])
        test_y  = np.array([te[i][1] for i in range(len(te))])

    elif dataset_name == "MNIST-SVHN-FASHION":
        # Exactly mirrors dataset.py concatenation order:
        #   MNIST (0-9) + SVHN (0-9) + FashionMNIST (10-19)
        mnist_tr  = datasets.MNIST(datadir, train=True, download=True)
        mnist_te  = datasets.MNIST(datadir, train=False, download=True)
        svhn_tr   = datasets.SVHN(datadir, split="train", download=True)
        svhn_te   = datasets.SVHN(datadir, split="test", download=True)
        fmnist_tr = datasets.FashionMNIST(datadir, train=True, download=True)
        fmnist_te = datasets.FashionMNIST(datadir, train=False, download=True)

        train_y = np.concatenate([
            np.array(mnist_tr.targets),                # 0-9
            np.array(svhn_tr.labels),                   # 0-9
            np.array(fmnist_tr.targets) + 10,           # 10-19
        ])
        test_y = np.concatenate([
            np.array(mnist_te.targets),
            np.array(svhn_te.labels),
            np.array(fmnist_te.targets) + 10,
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_y, test_y


def partition_classes_into_tasks(all_classes, num_tasks, rng):
    """Randomly partition *all_classes* into *num_tasks* groups that are
    as equal-sized as possible.  Returns list[task] -> list of classes."""
    perm = rng.permutation(all_classes)
    base, extra = divmod(len(perm), num_tasks)
    tasks_classes = []
    idx = 0
    for t in range(num_tasks):
        size = base + (1 if t < extra else 0)
        tasks_classes.append(sorted(perm[idx:idx + size].tolist()))
        idx += size
    return tasks_classes


def assign_client_classes(task_pool, cet, num_clients, rng):
    """For one task whose global class pool is *task_pool*, assign each
    client *cet* classes (subset of the pool).  If cet >= len(pool) every
    client simply gets the full pool."""
    pool = list(task_pool)
    result = []
    for _ in range(num_clients):
        if cet >= len(pool):
            result.append(sorted(pool))
        else:
            chosen = sorted(rng.choice(pool, size=cet, replace=False).tolist())
            result.append(chosen)
    return result


def collect_indices(labels_array, target_classes):
    """Return sorted list of indices where label is in target_classes."""
    mask = np.isin(labels_array, target_classes)
    return np.where(mask)[0].tolist()


# ------------------------------------------------------------------ #
#  Main generation logic                                              #
# ------------------------------------------------------------------ #
def generate_split(dataset_name, datadir):
    cfg = CONFIGS[dataset_name]
    num_clients = cfg["num_clients"]
    num_tasks   = cfg["num_tasks"]
    cet         = cfg["classes_per_task"]
    n_labels    = cfg["unique_labels"]
    seed        = cfg["seed"]
    filename    = cfg["filename"]

    rng = np.random.RandomState(seed)

    print(f"Loading labels for {dataset_name} ...")
    train_y, test_y = load_labels(dataset_name, datadir)
    print(f"  train samples: {len(train_y)}, test samples: {len(test_y)}")

    all_classes = list(range(n_labels))

    # 1. Partition classes into tasks
    tasks_classes = partition_classes_into_tasks(
        np.array(all_classes), num_tasks, rng)
    for t, cls in enumerate(tasks_classes):
        print(f"  Task {t} global pool: {cls}")

    # 2. For each task, assign each client a subset of cet classes
    client_y_list = [[] for _ in range(num_clients)]   # [client][task]
    for t in range(num_tasks):
        client_classes_t = assign_client_classes(
            tasks_classes[t], cet, num_clients, rng)
        for c in range(num_clients):
            client_y_list[c].append(client_classes_t[c])

    # 3. Build index lists
    train_inds = [[] for _ in range(num_clients)]
    test_inds  = [[] for _ in range(num_clients)]
    for c in range(num_clients):
        for t in range(num_tasks):
            cls = client_y_list[c][t]
            train_inds[c].append(collect_indices(train_y, cls))
            test_inds[c].append(collect_indices(test_y, cls))

    # 4. Verify
    for c in range(num_clients):
        for t in range(num_tasks):
            actual_train = set(train_y[np.array(train_inds[c][t])].tolist())
            expected     = set(client_y_list[c][t])
            assert actual_train == expected, \
                f"Mismatch client {c} task {t}: {actual_train} vs {expected}"

    # 5. Save
    out_dir = os.path.join(datadir, "data_split")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    split_data = {
        "train_inds":    train_inds,
        "test_inds":     test_inds,
        "client_y_list": client_y_list,
    }
    with open(out_path, "wb") as f:
        pickle.dump(split_data, f)

    print(f"\nSaved split to {out_path}")
    print(f"  {num_clients} clients, {num_tasks} tasks, {cet} classes/task")
    for c in range(min(3, num_clients)):
        for t in range(num_tasks):
            n_tr = len(train_inds[c][t])
            n_te = len(test_inds[c][t])
            print(f"    client {c} task {t}: classes={client_y_list[c][t]}  "
                  f"train={n_tr}  test={n_te}")
        if num_clients > 3:
            print("    ...")
            break

    return out_path


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate data-split .pkl files for AF-FCL")
    parser.add_argument("--dataset", type=str, default="MNIST-SVHN-FASHION",
                        choices=list(CONFIGS.keys()))
    parser.add_argument("--datadir", type=str, default="datasets/PreciseFCL/")
    args = parser.parse_args()
    generate_split(args.dataset, args.datadir)
