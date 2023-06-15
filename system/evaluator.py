import torch
import numpy as np
import random

from transformers import is_torch_available
from manager import BaseDataManager


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def compute_acc(generations, outputs, output2labels):
    num, acc = 0, 0
    for generation, output in zip(generations, outputs):
        if generation == output2labels[output]:
            acc += 1
        num += 1
    return acc / num * 100.0


def evaluate_unsupervised_generation(train: BaseDataManager):
    pass
