from .gcn_a import gcn_a

__factory__ = {
    'gcn_a': gcn_a,
}


def build_model(name, *args, **kwargs):
    if name not in __factory__:
        raise KeyError("Unknown model:", name)
    return __factory__[name](*args, **kwargs)
