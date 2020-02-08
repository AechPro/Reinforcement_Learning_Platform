from Experiments import ExperimentManager
import os

if __name__ == "__main__":
    path = os.path.join("resources", "experiments", "test_experiments", "test_experiment.json")
    manager = ExperimentManager()
    manager.load_experiment(path)
    manager.run_experiments()