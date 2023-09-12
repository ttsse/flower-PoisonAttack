"""A testing module to test the data load function."""
import pytest
import sys
import os
sys.path.append(os.path.abspath("src"))

from datasets import load_data

def get_input_combo():
    all_dataset_names = ["MNIST", "CIFAR-10", "EMNIST"]
    dataset_path = "./temp/data"
    for dataset_name in all_dataset_names:
            yield (dataset_name, dataset_path)

@pytest.mark.parametrize("dataset_name, dataset_path", [x for x in get_input_combo()])
def test_model_load(dataset_name, dataset_path):
    trainset, testset = load_data(
         dataset_name=dataset_name,
         dataset_path=dataset_path,
    )
    assert trainset is not None, "Something went wrong while loading data"
