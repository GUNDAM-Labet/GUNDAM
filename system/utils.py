import os
import numpy as np
import json

from huggingface_hub import hf_hub_download
from typing import Optional, Dict, Union
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
    priority_level: int = 0


@dataclass
class Dataset2Key:
    data_name: str
    dataset_name: str
    train_type: str
    valid_type: str
    source_key: str
    target_key: str
    converter_name: str


class ConfigData:   # for convenience of using huggingface datasets
    def __init__(self):
        self.dataset2keys = {}

    def load(self):    
        dataset2key_sst2 = Dataset2Key(
            data_name="sst2", 
            dataset_name="sst2",
            train_type="train",
            valid_type="validation",
            source_key="sentence", 
            target_key="label",
            converter_name="sst2"
        ) 
        self.add(dataset2keys=dataset2key_sst2)
        
        dataset2key_cola = Dataset2Key(
            data_name="cola",
            dataset_name="linxinyuan/cola",
            train_type="train",
            valid_type="test",
            source_key="text", 
            target_key="label",
            converter_name="cola"
        )
        self.add(dataset2key_cola)

        dataset2key_sst5 = Dataset2Key(
            data_name="sst5",
            dataset_name="SetFit/sst5",
            train_type="train",
            valid_type="test",
            source_key="text",
            target_key="label",
            converter_name="sst5"
        )
        self.add(dataset2key_sst5)

        dataset2key_fpb = Dataset2Key(
            data_name="fpb",
            dataset_name="cyrilzhang/financial_phrasebank_split",
            train_type="train",
            valid_type="test",
            source_key="sentence",
            target_key="label",
            converter_name="fpb"
        )
        self.add(dataset2key_fpb)

        dataset2key_trec = Dataset2Key(
            data_name="trec",
            dataset_name="trec",
            train_type="train",
            valid_type="test",
            source_key="text",
            target_key="coarse_label",
            converter_name="trec"
        )
        self.add(dataset2key_trec)

        dataset2key_subj = Dataset2Key(
            data_name="subj",
            dataset_name="SetFit/subj",
            train_type="train",
            valid_type="test",
            source_key="text",
            target_key="label",
            converter_name="subj"
        )
        self.add(dataset2key_subj)
    
    def add(self, dataset2keys: Union[Dict, Dataset2Key]):
        if isinstance(dataset2keys, Dict):
            self.dataset2keys.update(dataset2keys)
        elif isinstance(dataset2keys, Dataset2Key):
            self.dataset2keys[dataset2keys.data_name] = dataset2keys
    
    def get(self, data_name: str) -> Dataset2Key:
        if isinstance(self.dataset2keys[data_name], Dataset2Key):
            return self.dataset2keys[data_name]
        else:
            raise ValueError
        

class ConfigGenerator:
    def __init__(self, decode_method: str = "beam", add_score: bool = False, num_generate: int = 5, 
                    max_source_len: int = 64, max_target_len: int = 10, max_new_tokens: int = 150, 
                    num_batch: int = None, num_return_sequence: int = None, temperature: float = 1.0):
        self.decode_method = decode_method
        self.add_score = add_score
        self.num_generate = num_generate
        self.num_return_sequence = num_return_sequence
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.num_batch = num_batch
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def config(self, decode_method=None, add_score=None, num_generate=None, max_new_tokens=None, num_batch=None, 
                num_return_sequence=None, temperature=None, max_source_len=None, max_target_len=None):
        self.decode_method = decode_method if decode_method else self.decode_method
        self.add_score = add_score if add_score else self.add_score
        self.num_generate = num_generate if num_generate else self.num_generate
        self.num_return_sequence = num_return_sequence if num_return_sequence else self.num_return_sequence
        self.temperature = temperature if temperature else self.temperature
        self.max_new_tokens = max_new_tokens if max_new_tokens else self.max_new_tokens
        self.num_batch = num_batch if num_batch else self.num_batch
        self.max_source_len = max_source_len if max_source_len else self.max_source_len
        self.max_target_len if max_target_len else self.max_target_len
    
    def get(self):
        method_to_kwargs = {
            "beam": {
                "num_beams": self.num_generate,
                "early_stopping": True,
                "num_return_sequences": self.num_return_sequence if self.num_return_sequence else self.num_generate
            },
            "greedy": {
                "do_sample": False
            },
            "sample": {
                "do_sample": True,
                "num_return_sequences": self.num_return_sequence if self.num_return_sequence else self.num_generate
            },
            "greedy_add_score": {
                "return_dict_in_gen": True,
                "output_score": True
            }
        }
        common_kwargs = {
            "temperature": self.temperature
        }
        if self.add_score and self.decode_method == "greedy":
            self.decode_method = "greedy_add_score"
        kwargs = method_to_kwargs[self.decode_method]
        kwargs.update(common_kwargs)
        return kwargs


def load_openai_key(key_file: str = None):
    if not key_file:
        current_path = os.path.abspath(os.getcwd())
        key_file = os.path.join(os.path.dirname(current_path), "openai_key.json") 
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            openai_key = json.load(f)
        assert isinstance(openai_key, str), f"openai_key {type(openai_key)} must be str"
        return openai_key
    else:
        raise FileNotFoundError