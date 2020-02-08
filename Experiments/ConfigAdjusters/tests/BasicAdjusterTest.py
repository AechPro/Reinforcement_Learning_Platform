from Experiments.ConfigAdjusters import BasicAdjuster
from Config import ConfigLoader
import json

def run_test():
    experiment_json = json.load(open("resources/experiments/test_experiments/test_experiments.json", 'r'))
    cfg = ConfigLoader.load_config(file_name=experiment_json["config_file"])

    basic_adjuster = BasicAdjuster()
    basic_adjuster.init(experiment_json["config_adjustments"]["adjustment_0"], cfg)

    for i in range(1000000):
        basic_adjuster.adjust_config(cfg)
        done = basic_adjuster.step()

        print(cfg["policy"]["init_std"])
        if done:
            break