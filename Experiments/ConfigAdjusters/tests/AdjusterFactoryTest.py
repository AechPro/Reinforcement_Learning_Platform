from Experiments.ConfigAdjusters import AdjusterFactory
from Config import ConfigLoader
import json

def run_test():
    experiment_json = json.load(open("resources/experiments/test_experiments/test_experiments.json", 'r'))
    cfg = ConfigLoader.load_config(file_name=experiment_json["config_file"])

    adjusters = AdjusterFactory.build_adjusters_for_experiment(experiment_json["config_adjustments"], cfg)
    for adjuster in adjusters:
        print("OPERATING WITH ADJUSTER",type(adjuster))
        for i in range(1000000):
            adjuster.adjust_config(cfg)
            done = adjuster.step()

            print(cfg["policy"]["init_std"], cfg["gradient_optimizer"]["step_size"],
                  cfg["policy_optimizer"]["noise_std"], cfg["novelty"]["num_strategy_frames"])

            if done:
                adjuster.reset_config(cfg)
                break
