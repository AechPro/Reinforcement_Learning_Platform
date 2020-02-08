from Algorithms import OptimizerFactory

class OptimizationManager(object):
    def __init__(self, config):
        self.cfg = config
        self.optimizer = None

    def configure(self):
        self.optimizer = OptimizerFactory.get_from_config(self.cfg)
        self.optimizer.configure()

    def reconfigure(self):
        if self.optimizer is not None:
            self.optimizer.reconfigure()

    def step(self):
        if self.optimizer is not None:
            self.optimizer.step()

    def is_done(self):
        return False

    def set_base_dir(self, base_dir):
        pass

    def cleanup(self):
        if self.optimizer is not None:
            self.optimizer.cleanup()
        del self.optimizer