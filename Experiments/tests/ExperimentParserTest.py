import json

from Experiments import ExperimentParser


def run_test():
    experiment = json.load(open("resources/experiments/test_experiments/test_experiments.json","r"))
    collected_adjustments = ExperimentParser.parse_adjustments(experiment["config_adjustments"])
    for adjustment in collected_adjustments:
        print(adjustment)