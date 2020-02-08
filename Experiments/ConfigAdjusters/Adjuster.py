import os
class Adjuster(object):
    def __init__(self):
        self.adjustments = []

    def init(self, adjustments_json, cfg):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def adjust_config(self, cfg):
        raise NotImplementedError

    def reset_per_increment(self):
        raise NotImplementedError

    def get_name(self):
        name = ""
        if self.adjustments is not None:
            for adjustment in self.adjustments:
                #name = "{}_{}".format(name, adjustment.get_name())
                name = os.path.join(name, adjustment.get_name())
        if name[0] == "_":
            name = name[1:]
        return name

    def reset(self):
        if self.adjustments is None or len(self.adjustments) == 0:
            return

        for adjustment in self.adjustments:
            adjustment.reset()

    def reset_config(self, cfg):
        if self.adjustments is None or len(self.adjustments) == 0:
            return

        for adjustment in self.adjustments:
            adjustment.reset_config(cfg)