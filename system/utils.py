import numpy as np
import os
import json

from torch.utils.data import Dataset
from typing import List, Optional, Iterator, Dict
from dataclasses import dataclass

IO_SEP_TOKEN = "|||"
PAD_TOKEN = "<pad>"


@dataclass
class Unit:
    unit_id: str
    source_input: str
    target_output: str
    source_emb: Optional[np.ndarray] = None
    target_emb: Optional[np.ndarray] = None
    priority_level: int


def trim_batch_data(input_ids, pad_token_id, attention_mask=None): # remove columns by pad_token_id
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])
    else:
        return input_ids[:, keep_column_mask]    