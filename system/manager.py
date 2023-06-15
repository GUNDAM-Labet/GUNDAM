import numpy as np
import json
import os
import random
import openai

from tqdm import trange
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Iterator
from data import Unit
from retriever import BaseRetriever
from miner import BaseMiner
from generator import Generator


class BaseManager():
    def __init__(self, data_type: str, data_path: str, embed_path: str):
        self.data_type = data_type  # train or valid
        self.embed_path = embed_path    # vectorized data storage
        self.data_path = data_path  # text data storage
        self.data = {}
    
    def load(self, data_dict: Dict = None, source_key: str = "question", target_key: str = "answer"):
        if os.path.exists(self.data_path):
            self.data = self._load_from_stored_data()
        else:
            self.data = self._load_from_raw_data(data_dict=data_dict, source_key=source_key, target_key=target_key)        
        
        self._load_embedding()
    
    def shuffle(self):
        unit_list = list(self.data.values())
        random.shuffle(unit_list)
        self.data = {unit.unit_id: unit for unit in unit_list}

    def _load_from_stored_data(self) -> Dict[str, Unit]: # load from json
        with open(self.data_path, "r") as f:
            data = json.load(f)
        # convert dictionaries back to Unit objects
        data = {unit_dict["unit_id"]: Unit(**unit_dict) for unit_dict in data}
        return data

    def _load_from_raw_data(self, data_dict, source_key: str = "question", target_key: str = "answer") -> Dict[str, Unit]:
        res = {}

        # data_dict is not None, load from data_dict, otherwise try data_path
        if data_dict:
            for idx, dic in enumerate(data_dict[self.data_type]):
                uid = f"{self.data_type}-{idx}"
                res[uid] = Unit(
                    unit_id=uid,
                    source_input=dic[source_key],
                    target_emb=dic[target_key],
                    priority_level=0
                )
        else:
            with open(self.data_path, "r") as f:
                for idx, line in enumerate(f):
                    dic = json.loads(line)
                    idx = f"{self.data_type}_{idx}"
                    res[idx] = Unit(
                        unit_id=idx,
                        source_input=dic[source_key],
                        target_output=dic[target_key],
                        priority_level=0
                    )
        return res
      
    def _compute_embedding(self, batch_size: int = 128, model: str ="text-embedding-ada-002"):
        # choices for model = ["all-mpnet-base-v2", "text-embedding-ada-002"]
        if model.startswith("text-embedding-"):
            model_type = "openai"
        else:
            model_type = "sbert"

        model_dim = {"all-mpnet-base-v2": 768, "text-embedding-ada-002": 1536}[model]
        all_inputs = [unit.source_input for unit in self.data.values()] 
        x = np.zeros((len(all_inputs), model_dim), dtype=np.float32)

        for i in trange(0, len(all_inputs), batch_size):
            j = min(i + batch_size, len(all_inputs))
            if model_type == "openai":
                res = openai.Embedding.create(input=[s.replace("\n", " ") for s in all_inputs[i:j]], model=model)
                embeddings = [None] * len(all_inputs[i:j])
                for dic in res["data"]:
                    embeddings[dic["index"]] = dic["embedding"]
            elif model_type == "sbert":
                embeddings = SentenceTransformer(model, device="cuda").encode(all_inputs[i:j])
            else:
                raise NotImplementedError
            
            x[i:j] = np.array(embeddings, dtype=np.float32)
        
        embed_path = os.path.join(self.embed_path, f"{self.data_type}_source.npy")
        np.save(embed_path, x)

    def _load_embedding(self):
        if not embed_path:
            return
        embed_path = os.path.join(self.embed_path, f"{self.data_type}_source.npy")
        if not os.path.exists(embed_path):
            return
        embeddings = np.load(embed_path)
        
        source_ids = [idx.split("-")[1] for idx in self.data.keys()]
        source_emb = np.take(embeddings, source_ids, axis=0)

        for idx, unit in enumerate(self.data.values()):
            unit.source_emb = source_emb[idx]
            assert isinstance(unit.source_emb, np.ndarray), f"{type(unit.source_emb)}"

    def save(self):
        directory_path = os.path.dirname(self.data_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)

        data = [unit.__dict__ for unit in self.data.values()]
        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=4)
    
    def batch_data(self, batch_size: int) -> Iterator[List[Unit]]:
        batch = []
        for unit in self.data.values():
            batch.append(unit)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:   # remaining units in the case where total number is not a multiple of batch_size
            yield batch



class GUNDAMManager(BaseManager):
    def __init__(self, generator: Generator, miner: BaseMiner, retriever: BaseRetriever,
                    data_type: str, data_path: str, embed_path: str):
        super().__init__(data_type=data_type, data_path=data_path, embed_path=embed_path)

        self.generator = generator
        self.miner = miner
        self.retriever = retriever

    def _reset_priority(self):
        for unit in self.data.values():
            unit.priority_level = 0

    def _update_priority(self, golden_ids: List[str]):
        for idx in golden_ids:
            if idx in self.data:
                self.data[idx].priority_level += 1    
    
    def get_priority_data(self):
        data = {}
        priority_level = []

        for unit in self.data.values():
            if unit.priority_level not in priority_level:
                priority_level.append(unit.priority_level)
                data[f"{unit.priority_level}"] = []
            else:
                data[f"{unit.priority_level}"].append(unit)
        return data
    
    def update(self): # run miner
        data = self.get_priority_data()
        max_priority_level = max(data.keys())
        golden_ids = self.miner.mine(data[f"{max_priority_level}"])
        self._update_priority(golden_ids)

    def act(self, priority_level: int = 0):
        data = self.get_priority_data()
        max_priority_level = max(data.keys())
        assert priority_level > max_priority_level, f"priority_level {priority_level} is higher than max {max_priority_level}"
        data = data[f"{max_priority_level}"]


    def tune(self): # tune generator
        pass