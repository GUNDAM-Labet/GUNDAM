import numpy as np
import json
import os
import random

from typing import Dict
from data import Unit
from retriever import (
    HardRetriever, RandomRetriever, SimilarRetriever, DiverseRetriever
)
from miner import BasicMiner


class GUNDAMAgent():
    def __init__(self):
        self.data = self.load()

    def load(self):
        pass

    def shuffle(self):
        unit_list = list(self.data.values())
        random.shuffle(unit_list)
        self.data = {unit.unit_id: unit for unit in unit_list}

    def load_from_stored_data(self):
        pass


    @staticmethod
    def load_from_raw_data(data_type, source_key, target_key, data_path, data_dict) -> Dict[str, Unit]:
        res = {}
        # data_dict is not None, load from data_dict, otherwise try data_path
        if data_dict:
            for idx, dic in enumerate(data_dict[data_type]):
                idx = f"{data_type}_{idx}"
                res[idx] = Unit(
                    unit_id=idx,
                    source_input=dic[source_key],
                    target_output=dic[target_key],
                    priority_level=0,
                )
            return res
        if data_path:
            with open(data_path, "r") as fin:
                for idx, line in enumerate(fin):
                    dic = json.loads(line)
                    idx = f"{data_type}_{idx}"
                    res[idx] = Unit(
                        unit_id=idx,
                        source_input=dic[source_key],
                        target_output=dic[target_key],
                        priority_level=0
                    )
            return res
        return res
    

    @staticmethod
    def save_to_json(data_dict, data_path: str):
        directory_path = os.path.dirname(data_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        with open(data_path, "w") as fin:
            json.dump(data_dict, fin, indent=4)
    

    def _load_embedding(self, data_type, emb_path):
        if not emb_path:
            return
        emb_path = os.path.join(emb_path, f"{data_type}")

    def initial_priority(self):
        pass

    def update_priority(self):
        pass

    def act(self):
        pass

    def tune(self): # tune generator
        pass

    def update(self): # re-run miner
        pass
