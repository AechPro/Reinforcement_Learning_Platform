from Experiments.ConfigAdjusters import GridAdjuster
from Config import ConfigLoader
import json

def run_test():
    experiment_json = json.load(open("resources/experiments/test_experiments/test_experiments.json", 'r'))
    cfg = ConfigLoader.load_config(file_name=experiment_json["config_file"])

    grid_adjuster = GridAdjuster()
    grid_adjuster.init(experiment_json["config_adjustments"]["grid"], cfg)

    for i in range(1000000):
        grid_adjuster.adjust_config(cfg)
        done = grid_adjuster.step()

        print(cfg["policy"]["init_std"],cfg["gradient_optimizer"]["step_size"],
              cfg["policy_optimizer"]["noise_std"], cfg["novelty"]["num_strategy_frames"])
        if done:
            break