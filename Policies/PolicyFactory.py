def get_from_config(cfg):
    implementation_type = cfg["policy"]["type"][0]
    imp = implementation_type.strip().lower()

    #todo: Implement policies in keras/tensorflow and include here.
    if imp in ("torch", "pytorch"):

        policy_type = cfg["policy"]["type"][1]
        f = policy_type.strip().lower()
        if f in ("cnn", "conv", "atari", "convolutional"):
            from Policies.PyTorch import Convolutional
            return Convolutional

        if f in ("rnn", "recurrent", "rec", "lstm", "gru"):
            from Policies.PyTorch import Recurrent
            return Recurrent

        from Policies.PyTorch import FeedForward
        return FeedForward