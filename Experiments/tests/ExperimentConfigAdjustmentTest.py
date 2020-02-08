import json

from Config import ConfigLoader
from Experiments import ExperimentParser


def run_test():
    experiment = json.load(open("resources/experiments/test_experiments/test_experiments.json", "r"))
    collected_adjustments = ExperimentParser.parse_adjustments(experiment["config_adjustments"])

    cfg = ConfigLoader.load_config(file_name=experiment["config_file"])
    initial = cfg["gradient_optimizer"]["step_size"]
    for adjustment in collected_adjustments:
        cfg = adjustment.setup_initial_config_parameters(cfg)


    before = cfg["gradient_optimizer"]["step_size"]

    for _ in range(100):
        for adjustment in collected_adjustments:
            cfg, success = adjustment.get_adjusted_config(cfg)

    after = cfg["gradient_optimizer"]["step_size"]
    for adjustment in collected_adjustments:
        cfg = adjustment.reset_config_value(cfg)

    final = cfg["gradient_optimizer"]["step_size"]

    print("INITIAL: {} BEFORE: {} AFTER: {} INITIAL RESET: {}"
          .format(initial, before, after, final))