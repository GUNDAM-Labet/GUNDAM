import numpy as np
import json
import os
import random
import openai

from tqdm import trange
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Iterator, Union
from utils import Unit, Dataset2Key
from retriever import BaseRetriever, DiverseRetriever
from miner import BaseMiner
from generator import BaseGenerator
from utils import load_openai_key


class BaseManager():
    def __init__(self, data_type: str, data_path: str = None, embed_path: str = None, data_name: str = None, 
                    embed_model: str = None):
        self.data_type = data_type  # train or valid
        self.data_name = data_name  # name of dataset
        
        if not data_path:
            assert data_name, "at least one of data_path and data_name should be vaild"
            current_path = os.path.abspath(os.getcwd())
            self.data_path = os.path.join(os.path.dirname(current_path), "data/")
        else:
            self.data_path = data_path  # text data storage
        self.embed_path = embed_path if embed_path else self.data_path  # vectorized data storage
        
        self.embed_model = embed_model
        self.data = {}
    
    def __len__(self):
        return len(self.data)  
    
    def load(self, data_dict: Dict = None, source_key: str = None, target_key: str = None, re_compute: bool = True, load_embedding: bool = False):
        data_path = os.path.join(self.data_path, f"{self.data_name}_{self.data_type}.json")
        if os.path.exists(data_path):
            self.data = self._load_from_stored_data(data_path=data_path)
        else:
            assert data_dict, f"at least one of data_path and data_dict should be vaild"
            self.data = self._load_from_raw_data(data_dict=data_dict, source_key=source_key, target_key=target_key) 
        
        embed_path = os.path.join(self.embed_path, f"{self.data_name}_{self.data_type}.npy")

        if load_embedding:
            if os.path.exists(embed_path) and not re_compute:
                self._load_embedding(embed_path=embed_path)
            else:
                self._compute_embedding()
                self._load_embedding(embed_path=embed_path)

    def shuffle(self):
        unit_list = list(self.data.values())
        random.shuffle(unit_list)
        self.data = {unit.unit_id: unit for unit in unit_list}

    def _load_from_stored_data(self, data_path) -> Dict[str, Unit]: # load from json
        with open(data_path, "r") as f:
            data = json.load(f)
        # convert dictionaries back to Unit objects
        data = {unit_dict["unit_id"]: Unit(**unit_dict) for unit_dict in data}
        return data

    def _load_from_raw_data(self, data_dict: Dict, source_key: str, target_key: str) -> Dict[str, Unit]:
        data = {}
        # data_dict is not None, load from data_dict, otherwise try data_path
        for idx, dic in enumerate(data_dict[self.data_type]):
            uid = f"{self.data_type}-{idx}"
            data[uid] = Unit(
                unit_id=uid,
                source_input=dic[source_key],
                target_output=dic[target_key]                
            )
        return data
      
    def _compute_embedding(self, batch_size: int = 128):
        assert self.embed_model in ["all-mpnet-base-v2", "text-embedding-ada-002"], f"embed_model must be [all-mpnet-base-v2, text-embedding-ada-002]"
        if self.embed_model.startswith("text-embedding-"):
            model_type = "openai"
            openai.api_key = load_openai_key()
        else:
            model_type = "sbert"
        model_dim = {"all-mpnet-base-v2": 768, "text-embedding-ada-002": 1536}[self.embed_model]
        all_inputs = [unit.source_input for unit in self.data.values()] 
        x = np.zeros((len(all_inputs), model_dim), dtype=np.float32)

        for i in trange(0, len(all_inputs), batch_size):
            j = min(i + batch_size, len(all_inputs))
            if model_type == "openai":
                res = openai.Embedding.create(input=[s.replace("\n", " ") for s in all_inputs[i:j]], model=self.embed_model)
                embeddings = [None] * len(all_inputs[i:j])
                for dic in res["data"]:
                    embeddings[dic["index"]] = dic["embedding"]
            elif model_type == "sbert":
                embeddings = SentenceTransformer(self.embed_model, device="cuda").encode(all_inputs[i:j])
            else:
                raise NotImplementedError
            
            x[i:j] = np.array(embeddings, dtype=np.float32)
            self._save_embedding(embedding=x)
    
    def _save_embedding(self, embedding: np.ndarray):
        embed_path = os.path.join(self.embed_path, f"{self.data_name}_{self.data_type}.npy")
        np.save(embed_path, embedding)

    def _load_embedding(self, embed_path: str):
        embeddings = np.load(embed_path)
        
        source_ids = [idx.split("-")[1] for idx in self.data.keys()]
        source_emb = np.take(embeddings, source_ids, axis=0)

        for idx, unit in enumerate(self.data.values()):
            unit.source_emb = source_emb[idx]
            assert isinstance(unit.source_emb, np.ndarray), f"{type(unit.source_emb)}"
        
    def _clear_embedding(self):
        for unit in self.data.values():
            unit.source_emb = None
            unit.target_emb = None

    def save(self):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path, exist_ok=True)
        
        data_path = os.path.join(self.data_path, f"{self.data_name}_{self.data_type}.json")
        self._clear_embedding()
        data = [unit.__dict__ for unit in self.data.values()]
        with open(data_path, "w") as f:
            json.dump(data, f, indent=4)
    
    def batch(self, batch_size: int) -> Iterator[List[Unit]]:
        batch = []
        for unit in self.data.values():
            batch.append(unit)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:   # remaining units in the case where total number is not a multiple of batch_size
            yield batch


class GUNDAMManager(BaseManager):
    def __init__(self, data_type: str, data_path: str = None, embed_path: str = None, data_name: str = "cola", 
                    embed_model: str = "all-mpnet-base-v2", generator: BaseGenerator = None, 
                    miner: BaseMiner = None, retriever: BaseRetriever = None):
        super().__init__(data_type=data_type, data_path=data_path, embed_path=embed_path, data_name=data_name, embed_model=embed_model)
        self.generator = generator
        self.miner = miner
        self.retriever = retriever
        self.converter = miner.converter if self.miner else None      

    def check(self):
        assert isinstance(self.generator, BaseGenerator) 
        assert isinstance(self.miner, BaseMiner) 
        assert isinstance(self.retriever, BaseRetriever)
        if not self.miner.generator:    # if miner.generator is None, use generator4evaluation as generator4miner
            self.miner.generator = self.generator

    def set_retriever(self, priority_level: int = 0, n_shots: int = 2):
        data, max_priority_level = self.get_priority_data()
        assert priority_level <= max_priority_level, f"priority_level {priority_level} is higher than max {max_priority_level}"
        self.retriever.n_shots = n_shots
        self.retriever.set_retrieval_pool(list(data[f"{priority_level}"].values()))
    
    def set_generator(self, cfg):
        self.generator.cfg = cfg
    
    def set_miner(self, cfg):
        self.miner.generator.cfg = cfg

    def _reset_priority(self, priority_level=0):
        assert priority_level >= 0, f"invaild priority level {priority_level}"
        for unit in self.data.values():
            if unit.priority_level > priority_level:
                unit.priority_level = priority_level

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
                data[f"{unit.priority_level}"] = {}
                data[f"{unit.priority_level}"].update({unit.unit_id: unit})
            else:
                data[f"{unit.priority_level}"].update({unit.unit_id: unit})
        return (data, max(priority_level))
    
    def update(self): # run miner
        data, max_priority_level = self.get_priority_data()
        golden_ids = self.miner.mine(data[f"{max_priority_level}"])
        if golden_ids:
            self._update_priority(golden_ids)
        else:
            print(f"can not update priority, max priority level is {max_priority_level}")

    def act(self, idx, batch):
        if isinstance(self.retriever, DiverseRetriever):
            batch_size = self.generator.batch_size
            batch_samples = self.retriever.get_batch_samples(target_indices=list(range(idx*batch_size, idx*batch_size+len(batch))))
        else:
            batch_samples = self.retriever.get_batch_samples(targets=batch)
        inputs = [self.converter.unit2code(units=samples, target=unit) for samples, unit in zip(batch_samples, batch)]
        generations = self.generator.act(input_text=inputs)
        return generations

    def split_and_convert(self, split_ratio: float):
        assert split_ratio < 1, f"split_ratio {split_ratio} should be smaller than 1"
        uids = list(self.data.keys())
        train_ids = random.choices(uids, k=int(split_ratio*len(uids)))
        train, valid = {}, {}
        
        for uid in uids:
            if uid in train_ids:
                train[uid] = Unit(
                    unit_id = uid,
                    source_input=self.data[uid].source_input,
                    target_output=self.converter.OUTPUT2LABEL[self.data[uid].target_output]
                )
            else:
                valid[uid] = Unit(
                    unit_id=uid,
                    source_input=self.data[uid].source_input,
                    target_output=self.converter.OUTPUT2LABEL[self.data[uid].target_output]
                )
        assert len(train) + len(valid) == len(self.data)
        return (train, valid)

    def tune(self, num_epoch: int = 1, split_ratio: float = 0.8, reset_priority_level: bool = False, priority_level: int = 0): # tune generator
        train, valid = self.split_and_convert(split_ratio=split_ratio)
        self.generator.tune(train=train, valid=valid, num_epoch=num_epoch)
        if reset_priority_level:
            self._reset_priority(priority_level=priority_level)




# ===== DEBUG =====
if __name__ == "__main__":
    import datasets
    from utils import ConfigData
    data_name = "cola"
    cfg = ConfigData()
    cfg.load()

    dataset = datasets.load_dataset(cfg.get(data_name).dataset_name)

    manager = GUNDAMManager(data_type="train", embed_model="text-embedding-ada-002")
    manager.load(data_dict=dataset, source_key=cfg.get(data_name).source_key, target_key=cfg.get(data_name).target_key, re_compute=False)
    manager.shuffle()
    manager.save()

    from retriever import HardRetriever, SimilarRetriever, RandomRetriever
    from miner import One2OneMiner
    from converter import SST2Converter
    from generator import GPTGenerator
    retriever = RandomRetriever()
    manager.retriever = retriever
    manager.set_retriever()

    converter = SST2Converter()
    miner = One2OneMiner()
    print("=====0=====")
    generator = GPTGenerator(model_name="gpt2-medium")
    generator.batch_size = 8
    print("=====1=====")
    generator.load()
    miner.converter = converter
    print("=====2=====")
    miner.generator = generator
    print("=====3=====")
    manager.generator = generator
    print("=====4=====")
    manager.miner = miner
    manager.converter = converter
    manager.tune()
    exit()
    manager.check()
    print("=====5=====")
    manager.update()
    print("=====6=====")
