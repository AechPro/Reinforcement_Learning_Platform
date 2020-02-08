from Config import ConfigLoader
from Experiments.ConfigAdjusters import AdjusterFactory
from Networking import ServerMessageHandler
import os
import numpy as np

class Experiment(object):
    def __init__(self, experiment_json, optimization_manager):
        self.experiment_json = experiment_json

        self.config_adjusters = None
        self.optimization_manager = optimization_manager
        self.cfg = None

        self.current_adjuster_index = 0

        self.num_trials = experiment_json["num_trials_per_adjustment"]
        self.terminal_conditions = experiment_json["terminal_conditions"]

        self.current_trial = 0

        self.base_dir = os.path.join(os.getcwd(),"data", "experiments")
        self.experiment_name = experiment_json["experiment_name"]
        self.adjustment_dir = ""

    def init(self):
        self.cfg = ConfigLoader.load_config(file_name=self.experiment_json["config_file"])

        self.config_adjusters = AdjusterFactory. \
            build_adjusters_for_experiment(self.experiment_json["config_adjustments"], self.cfg)

        self.config_adjusters[self.current_adjuster_index].reset_config(self.cfg)
        self.config_adjusters[self.current_adjuster_index].adjust_config(self.cfg)
        self.adjustment_dir = self.config_adjusters[self.current_adjuster_index].get_name()
        self.start_trial()

    def step(self):
        if self.is_done():
            return True

        self.optimization_manager.step()
        if self.optimization_manager.is_done():
            self.current_trial += 1
            self.next_trial()

        return False

    def get_next_adjustment(self):
        if self.is_done():
            return

        idx = self.current_adjuster_index
        self.config_adjusters[idx].adjust_config(self.cfg)
        self.adjustment_dir = self.config_adjusters[self.current_adjuster_index].get_name()

        done = self.config_adjusters[idx].step()

        if done:
            self.config_adjusters[idx].reset_config(self.cfg)
            self.current_adjuster_index += 1

    def next_trial(self):
        if self.current_trial >= self.num_trials:
            print("new adjustment")
            self.current_trial = 0
            self.get_next_adjustment()

            if self.is_done():
                print("experiment complete")
                self.optimization_manager.reset()
                return

            elif self.config_adjusters[self.current_adjuster_index].reset_per_increment():
                self.optimization_manager.reset()

            else:
                self.optimization_manager.reconfigure()

        else:
            self.optimization_manager.reconfigure()

        print("starting trial")
        self.start_trial()

    def start_trial(self):
        current_trial_dir = os.path.join(self.base_dir, self.experiment_json["experiment_name"],
                                                 self.adjustment_dir, str(self.current_trial))

        experiment_name = "{}_{}_{}".format(self.experiment_json["experiment_name"],
                                            self.adjustment_dir, self.current_trial)

        self.cfg[ServerMessageHandler.EXPERIMENT_ID_KEY] = experiment_name

        self.cfg["rng"] = np.random.RandomState(int(self.cfg["seed"]))

        print("set base dir")
        self.optimization_manager.set_base_dir(current_trial_dir)
        print("set conditions")
        self.optimization_manager.set_terminal_conditions(self.terminal_conditions)
        print("configure")
        self.optimization_manager.configure(self.cfg)
        print("trial started")

    def is_done(self):
        return self.current_adjuster_index >= len(self.config_adjusters)