"""Shared neural capacity settings without importing a training runtime."""
DEFAULT_HIDDEN = 512
DEFAULT_HIDDEN_LAYERS = 4


def hidden_sizes(hidden=DEFAULT_HIDDEN, hidden_layers=DEFAULT_HIDDEN_LAYERS):
    if hidden < 2 or hidden_layers < 1:
        raise ValueError('Require hidden >= 2 and hidden_layers >= 1')
    return tuple(max(2, hidden // (2 ** i)) for i in range(hidden_layers))
