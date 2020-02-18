import numpy as np
import torch

from Policies import Policy
import os

class TorchPolicy(Policy):
    def __init__(self, input_shape, output_shape, action_parser, cfg, cfg_key):
        super().__init__(input_shape, output_shape, action_parser, cfg, cfg_key)

    def build_model(self, model_instructions):
        raise NotImplementedError

    def activate(self, target_input, input_normalization=None):
        model_input = target_input

        if input_normalization is not None:
            mean, std = input_normalization
            model_input = np.subtract(target_input, mean) / std

        reshaped_input = self.reshape_input(model_input)
        model_input = torch.as_tensor(reshaped_input)
        output = self.model.forward(model_input)
        preds = np.asarray(output.data)

        del output
        del reshaped_input
        del target_input
        del model_input

        return preds

    def activate_batch(self, input_batch, input_normalization=None):
        return self.activate(input_batch, input_normalization=input_normalization)

    def compute_virtual_normalization(self, virtual_batch):
        self.model.train()
        self.activate(virtual_batch)
        self.model.eval()

        self.update_internal_bn_stats()

    def update_internal_flat(self):
        flat = []

        for layer in self.model:
            if not hasattr(layer, 'weight'):
                continue

            weight = np.ravel(layer.weight.data)
            bias = np.ravel(layer.bias.data)

            for arg in weight:
                flat.append(arg)
            for arg in bias:
                flat.append(arg)

            del weight
            del bias

        if self.num_params != len(flat):
            self.num_params = len(flat)

        np_flat = np.asarray(flat)
        del flat
        del self.flat

        self.flat = np_flat.astype("float32")
        del np_flat

    def get_trainable_flat(self, force_update=False):
        if self.flat is None or force_update:
            self.update_internal_flat()
        return self.flat

    def set_trainable_flat(self, flat):
        offset = 0

        for layer in self.model:
            if not hasattr(layer, 'weight'):
                continue

            weight_size = np.prod(layer.weight.size())
            bias_size = np.prod(layer.bias.size())

            step = weight_size + bias_size
            slice = flat[offset:offset + step]
            offset += step

            new_weight = slice[:weight_size]
            new_bias = slice[weight_size:]

            layer.weight.data = torch.as_tensor(np.reshape(new_weight, layer.weight.size()))
            layer.bias.data = torch.as_tensor(np.reshape(new_bias, layer.bias.size()))

            del new_weight
            del new_bias
            del slice

        self.update_internal_flat()

    def update_internal_bn_stats(self):
        bn_stats = []

        for layer in self.model:
            if hasattr(layer, "track_running_stats"):

                mean = []
                for arg in layer.running_mean:
                    mean.append(arg.item())

                var = []
                for arg in layer.running_var:
                    var.append(arg.item())

                bn_stats.append((mean, var, int(layer.num_batches_tracked.data)))

        self.bn_stats = bn_stats
        del bn_stats

    def get_bn_stats(self):
        if self.bn_stats is None:
            return []

        return self.bn_stats

    def set_bn_stats(self, stats_list):
        idx = 0
        for layer in self.model:
            if hasattr(layer, "track_running_stats"):
                mean, var, num = stats_list[idx]
                layer.running_mean.data[:] = torch.as_tensor(mean)
                layer.running_var.data[:] = torch.as_tensor(var)
                layer.num_batches_tracked -= layer.num_batches_tracked
                layer.num_batches_tracked += num
                idx += 1
        self.update_internal_bn_stats()
        
    def copy_from(self, other):
        flat = other.get_trainable_flat()
        bn = other.get_bn_stats()
        
        if len(bn) > 0:
            self.set_bn_stats(bn)
        self.set_trainable_flat(flat)
    
    def save(self, file_path):
        if self.flat is None:
            self.update_internal_flat()
        np.save(file_path, self.flat)

        if len(self.get_bn_stats()) == 0:
            return

        bn_stats = ""
        for layer in self.model:
            if hasattr(layer, "track_running_stats"):
                stats = ""

                for arg in layer.running_mean:
                    stats = "{},{}".format(stats, arg.item())

                stats = "{}_".format(stats)

                for arg in layer.running_var:
                    stats = "{},{}".format(stats, arg.item())

                stats = "{}_{}".format(stats, layer.num_batches_tracked.data)
                stats = stats.replace("_,", "_")

                bn_stats = "{}\n{}".format(bn_stats, stats[1:])

        bn_stats = bn_stats[1:]

        with open("{}/bn.dat".format(file_path[:file_path.rfind("/")]), "w") as f:
            f.write(bn_stats)


    def load(self, file_path):
        flat = np.load("{}.npy".format(file_path))
        self.set_trainable_flat(flat)

        bn_file = "{}/bn.dat".format(file_path[:file_path.rfind("/")])
        if os.path.exists(bn_file):
            bn_stats = []
            with open(bn_file, "r") as f:
                for line in f:
                    entries = line.split("_")
                    mean = entries[0].split(",")
                    var = entries[1].split(",")

                    num = int(entries[2].strip())
                    mean = [float(arg.strip()) for arg in mean]
                    var = [float(arg.strip()) for arg in var]
                    bn_stats.append((mean, var, num))

            self.set_bn_stats(bn_stats)

        self.model.eval()

    def copy(self, other):
        self.set_trainable_flat(other.get_trainable_flat())

    def cleanup(self):
        del self.model
        del self.flat