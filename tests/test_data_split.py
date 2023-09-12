"""A testing module to test the data split function."""
import pytest
import sys
import os
sys.path.append(os.path.abspath("src"))

from datasets import load_and_fetch_split

def get_input_combo():
    all_dataset_names = ["MNIST", "CIFAR-10", "EMNIST"]
    dataset_path = "./temp/data"
    split = True
    dirichlet_alphas = [0.1, 1.0, 100.0]
    random_seeds = [10, 32]
    n_clients = [10]

    for dataset_name in all_dataset_names:
        for alpha in dirichlet_alphas:
            for rand_seed in random_seeds:
                for n_client in n_clients:
                    yield (dataset_name, dataset_path, split, alpha, rand_seed, n_client)

@pytest.mark.parametrize("dataset_name, dataset_path, split, alpha, rand_seed, n_client", [x for x in get_input_combo()])
def test_model_load(dataset_name, dataset_path, split, alpha, rand_seed, n_client):
    data_configs = {
        "DATASET_NAME": dataset_name,
        "DATASET_PATH": dataset_path,
        "SPLIT": split,
        "DIRICHLET_ALPHA": alpha,
        "RANDOM_SEED": rand_seed,
    }
    trainset, testset = load_and_fetch_split(
        client_id = 0,
        n_clients = n_client,
        dataset_conf=data_configs,
    )
    assert trainset is not None, "Something went wrong while loading and splitting the dataset"
