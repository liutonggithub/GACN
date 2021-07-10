from .dataset import Dataset
from .build_dataloader import build_dataloader


def build_dataset(cfg):
    return Dataset(cfg)
