from .test_gcns import test_gcns
from .train_gcns import train_gcns

__factory__ = {
    'test_gcns': test_gcns,
    'train_gcns': train_gcns,
}


def build_handler(phase):
    key_handler = '{}_gcns'.format(phase)
    if key_handler not in __factory__:
        raise KeyError("Unknown op:", key_handler)
    return __factory__[key_handler]
