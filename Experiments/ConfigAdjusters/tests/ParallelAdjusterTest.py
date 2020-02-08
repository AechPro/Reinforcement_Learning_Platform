from Experiments.ConfigAdjusters import ParallelAdjuster
from Config import ConfigLoader
import json

def run_test():
    experiment_json = json.load(open("resources/experiments/test_experiments/test_experiments.json", 'r'))
    cfg = ConfigLoader.load_config(file_name=experiment_json["config_file"])

    parallel_adjuster = ParallelAdjuster()
    parallel_adjuster.init(experiment_json["config_adjustments"]["parallel"], cfg)

    for i in range(1000000):
        parallel_adjuster.adjust_config(cfg)
        done = parallel_adjuster.step()

        print(cfg["policy"]["init_std"],cfg["gradient_optimizer"]["step_size"],
              cfg["policy_optimizer"]["noise_std"], cfg["novelty"]["num_strategy_frames"])
        if done:
            break