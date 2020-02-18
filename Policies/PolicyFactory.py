from Policies import PolicyActionParsers

def get_from_config(cfg):
    models = {}
    action_parsers = {}
    if "value_estimator" in cfg.keys():
        models["value_estimator"] = _get_model(cfg["value_estimator"]["type"][0], cfg["value_estimator"]["type"][1])
        action_parsers["value_estimator"] = _get_action_parser(cfg["value_estimator"]["action_parser"])
        
    if "quality_estimator" in cfg.keys():
        models["quality_estimator"] = _get_model(cfg["quality_estimator"]["type"][0], cfg["quality_estimator"]["type"][1])
        action_parsers["quality_estimator"] = _get_action_parser(cfg["quality_estimator"]["action_parser"])
        
    if "policy" in cfg.keys():
        models["policy"] = _get_model(cfg["policy"]["type"][0], cfg["policy"]["type"][1])
        action_parsers["policy"] = _get_action_parser(cfg["policy"]["action_parser"])

    return models, action_parsers

def _get_model(implementation_type, model_type):
    imp = implementation_type.strip().lower()

    # todo: Implement policies in keras/tensorflow and include here.
    if imp in ("torch", "pytorch"):

        f = model_type.strip().lower()
        if f in ("cnn", "conv", "atari", "convolutional"):
            from Policies.PyTorch import Convolutional
            return Convolutional

        if f in ("rnn", "recurrent", "rec", "lstm", "gru"):
            from Policies.PyTorch import Recurrent
            return Recurrent

        from Policies.PyTorch import FeedForward
        return FeedForward

def _get_action_parser(parser_str):
    p = parser_str.strip().lower()
    if p in ("sample", "sampling", "choice", "probabilistic", "categorical"):
        return PolicyActionParsers.random_sample
    return PolicyActionParsers.linear_parse