from Experiments.ConfigAdjusters import Adjuster
from Experiments.ConfigAdjusters import Adjustment

class GridAdjuster(Adjuster):
    def __init__(self):
        super().__init__()
        self.current_adjustment_target = 0
        self.reset_this_increment = False

    def init(self, adjustments_json, cfg):
        for key, item in adjustments_json.items():
            if "adjustment" in key:
                adjustment = Adjustment()
                adjustment.init(item, cfg)
                self.adjustments.append(adjustment)

    def step(self):
        self.reset_this_increment = False
        return self.step_grid()

    def adjust_config(self, cfg):
        for adjustment in self.adjustments:
            adjustment.adjust_config(cfg)

    def reset_per_increment(self):
        return self.reset_this_increment

    def step_grid(self):
        done = True
        for adjustment in self.adjustments:
            if not adjustment.is_done():
                done = False

        if done:
            return True

        idx = self.current_adjustment_target

        zeroeth_reset = self.adjustments[0].is_done()
        while self.adjustments[idx].is_done():
            self.adjustments[idx].reset()

            self.current_adjustment_target += 1
            idx += 1

            if idx >= len(self.adjustments):
                self.current_adjustment_target = 0
                return False

            if idx != 0:
                if self.adjustments[idx].reset_per_increment:
                    self.reset_this_increment = True
                self.adjustments[idx].step()

        self.current_adjustment_target = 0

        if not zeroeth_reset:
            if self.adjustments[0].reset_per_increment:
                self.reset_this_increment = True
            self.adjustments[0].step()

        return False

