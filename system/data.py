import numpy as np
import os
import json

from torch.utils.data import Dataset
from typing import List, Optional, Iterator, Dict
from dataclasses import dataclass


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


class Seq2SeqDataset(Dataset):
    def __init__(self, tokenizer, data_dir, max_source_len, max_target_len, 
                    data_type="train", source_key="question", target_key="answer", num_obs=None, prefix=""):
        super().__init__()
        data_path = os.path.join(data_dir, f"{data_type}.jsonl")
        if data_type == "val" and not os.path.exists(data_path):
            data_path = os.path.join(data_dir, f"dev.jsonl")    # both val and dev are names for validation data
        
        self.src_arr = self.read_jsonl(data_path, field=source_key)
        self.tgt_arr = self.read_jsonl(data_path, field=target_key)
        self.src_lens = self.get_lens(self.src_arr)
        self.tgt_lens = self.get_lens(self.tgt_arr)
        self.max_src_len = max_source_len
        self.max_tgt_len = max_target_len
        assert min(self.src_lens) > 0, f"find empty line in {data_path}"

        self.tokenizer = tokenizer
        self.prefix = prefix
        if num_obs:
            self.src_lens = self.src_lens[:num_obs]        


    @staticmethod
    def read_jsonl(data_path, field):
        data = []
        with open(data_path) as f:
            for line in f:
                j = json.loads(line)
                data.append(j[field])
        return data
    

    @staticmethod
    def get_lens(arr):
        return [len(x) for x in arr]



class AutoregLMDataset(Dataset):
    def __init__(self, tokenizer, data_dir, max_source_len: int = 512, max_target_len: int = 128, 
                    data_type="train", source_key="question", target_key="answer", prefix=""):
        super().__init__()
        data_path = os.path.join(data_dir, f"{data_type}.jsonl")

        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.prefix = prefix

        self.IO_SEP = "|||"
        self.io_sep_token_id = self.tokenizer(self.IO_SEP)["input_idx"]
