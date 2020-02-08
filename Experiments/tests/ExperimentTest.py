import json

from Experiments import Experiment, ExperimentManager
from PolicyOptimizers import OptimizationManager
import os

class FakeOptimizer(object):
    def __init__(self):
        self.current_step = 0
        self.name = None

    def step(self):
        self.current_step += 1

    def is_done(self):
        return self.current_step >= 10

    def set_terminal_conditions(self, conditions):
        pass

    def set_base_dir(self, base_dir):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        idx = base_dir.find("test_experiment") + len("test_experiment")
        self.name = base_dir[idx:]

    def configure(self, cfg):
        print("***********CONFIGURING OPTIMIZER***********")
        print("NAME:", self.name)
        print(cfg["gradient_optimizer"]["min_dynamic_update_size"],
              cfg["gradient_optimizer"]["max_dynamic_update_size"],
              cfg["novelty"]["phi"])


    def reconfigure(self):
        self.current_step = 0

    def stop(self):
        pass

    def restart(self):
        pass

    def cleanup(self):
        pass

def run_test():
    experiment_json = json.load(open("resources/experiments/test_experiments/test_experiment.json", "r"))
    experiment = Experiment(experiment_json, FakeOptimizer())
    experiment.init()

    while not experiment.step():
        continue