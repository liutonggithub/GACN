from .test_gcn_a import test_gcn_v
from .train_gcn_a import train_gcn_a

__factory__ = {
    'test_gcn_a': test_gcn_a,
    'train_gcn_a': train_gcn_a,
}


def build_handler(phase, model):
    key_handler = '{}_{}'.format(phase, model)
    if key_handler not in __factory__:
        raise KeyError("Unknown op:", key_handler)
    return __factory__[key_handler]
