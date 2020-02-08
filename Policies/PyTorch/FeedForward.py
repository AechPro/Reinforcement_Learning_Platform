import numpy as np
import torch.nn

from Policies.PyTorch import TorchPolicy
from Util import TorchJSONParser


class FeedForward(TorchPolicy):
    def __init__(self, input_shape, output_shape, action_parser, cfg):
        super().__init__(input_shape, output_shape, action_parser, cfg)

    def build_model(self, model_instructions):
        layers = []

        for i in range(0, len(model_instructions["layers"])):
            if i > 0:
                prev_features = model_instructions["layers"][i - 1]
            else:
                prev_features = np.prod(self.input_shape)

            if model_instructions["layer_extras"] is not None and len(model_instructions["layer_extras"]) > i:
                processing_function, needs_features = TorchJSONParser.parse_function(model_instructions["layer_extras"][i])
                if processing_function is not None:
                    if needs_features:
                        layers.append(processing_function(num_features=prev_features))
                    else:
                        layers.append(processing_function())

            layer = torch.nn.Linear(in_features=prev_features,
                                    out_features=model_instructions["layers"][i])
            layers.append(layer)

            function, _ = TorchJSONParser.parse_function(model_instructions["layer_functions"][i])
            if function is not None:
                layers.append(function())

        prev_features = self.input_shape
        if len(model_instructions["layers"]) > 0:
            prev_features = model_instructions["layers"][-1]

        processing_function, needs_features = TorchJSONParser.parse_function(model_instructions["output_extras"])
        if processing_function is not None:
            if needs_features:
                layers.append(processing_function(num_features=prev_features))
            else:
                layers.append(processing_function())

        output_features = np.prod((self.output_shape,))
        layer = torch.nn.Linear(in_features=prev_features,
                                out_features=output_features)
        layers.append(layer)

        function, _ = TorchJSONParser.parse_function(model_instructions["output_function"])
        if function is not None:
            if function == torch.nn.Softmax:
                layers.append(function(dim=1))
            else:
                layers.append(function())

        del self.model
        self.model = torch.nn.Sequential(*layers)
        self.model.eval()
        self.update_internal_flat()
        self.update_internal_bn_stats()
        del layers
        del model_instructions
