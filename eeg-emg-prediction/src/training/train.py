import copy

from src.config import validate
from src.training.run_pipeline import output_guard, train_experiment


def train(dataset, config, name=None, progress=None, should_stop=None):
    c = copy.deepcopy(validate(config))
    return train_experiment(dataset, c, name=name, progress=progress, should_stop=should_stop)

