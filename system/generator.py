import torch

from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, AutoConfig
)
import logging

logging.basicConfig(level=logging.INFO)


class Generator:
    def __init__(self) -> None:
        pass