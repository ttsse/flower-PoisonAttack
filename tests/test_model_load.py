"""A testing module to test the data load function."""
import pytest
import sys
import os
sys.path.append(os.path.abspath("src"))

from models import load_model

def get_input_combo():
    all_model_names = ["SIMPLE-CNN", "SIMPLE-MLP"]
    all_num_classes = [x for x in range(1, 10)]
    for model_name in all_model_names:
        for num_class in all_num_classes:
            yield (model_name, num_class)

@pytest.mark.parametrize("model_name, num_class", [x for x in get_input_combo()])
def test_model_load(model_name, num_class):
    model_configs = {
        "MODEL_NAME": model_name,
        "NUM_CLASSES": num_class,
    }
    test_model = load_model(model_configs=model_configs)
    assert test_model is not None, "Something went wrong while loading the model."
