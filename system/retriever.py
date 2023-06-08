import multiprocessing as mp
import random
import numpy as np

from base import Unit
from typing import List
from tqdm import tqdm


class BaseRetriever:
    def __init__(self, n_process: int = 1, n_shots: int = 5):
        self.n_process = n_process
        self.n_shots = n_shots
    
    def get_samples(self, target: Unit) -> List[Unit]:
        raise NotImplementedError

    def get_batch_samples(self, targets: List[Unit]) -> List[Unit]:
        if self.n_process <= 1:
            return [self.get_samples(target) for target in targets]

        with mp.Pool(self.n_process) as pool:
            return list(
                tqdm(
                    pool.imap(self.get_samples, targets),
                    disable=False, total=len(targets)
                )
            )
    

class HardRetriever(BaseRetriever):
    def __init__(self, units: List[Unit], n_process: int = 1, n_shots: int = 5):
        super().__init__(n_process=n_process, n_shots=n_shots)
        assert isinstance(units, list)
        assert len(units) >= n_shots
    
        self.id2unit = {unit.unit_id: unit for unit in units}
    
    def get_samples(self, sample_ids: List[str], target: Unit) -> List[Unit]:

        samples = [self.id2unit[sample_id] for sample_id in sample_ids if sample_id in self.id2unit]
        assert len(samples) == self.n_shots
        return samples


class RandomRetriever(BaseRetriever):
    def __init__(self, units: List[Unit], n_process: int = 1, n_shots: int = 5):
        super().__init__(n_process=n_process, n_shots=n_shots)
        assert isinstance(units, list)
        assert len(units) >= n_shots

        self.id2unit = {unit.unit_id: unit for unit in units}

    def get_samples(self, target: Unit) -> List[Unit]:
        
        samples = random.sample(list(self.id2unit.values()), self.n_shots)
        return samples


class SimilarRetriever(BaseRetriever): # retriever top-K cosine similar units 
    def __init__(self, units: List[Unit], n_process: int = 1, n_shots: int = 5, use_norm: bool = False):
        super().__init__(n_process=n_process, n_shots=n_shots)
        assert isinstance(units, list)
        assert len(units) >= n_shots

        self.units = units
        self.instance_emb = np.array([unit.source_emb for unit in units], dtype=np.float64)
        if use_norm:
            u, s, vt = np.linalg.svd(self.instance_emb, full_matrices=False)
            self.W_norm = vt.T.dot(np.diag(1 / s)).dot(vt)
        
    def get_samples(self, target: Unit) -> List[Unit]:
        assert isinstance(target.source_emb, np.ndarray), f"{type(target.source_emb)}"
        x = self.instance_emb
        y = target.source_emb
        
        if hasattr(self, "W_norm"):
            x = x.dot(self.W_norm)
            y = y.dot(self.W_norm)
        
        dist = -x.dot(y) / np.sqrt((x ** 2).sum(1))
        sample_ids = np.argsort(dist)[:self.n_shots]
        samples = [self.units[sample_id] for sample_id in sample_ids[::-1]]
        return samples


class DiverseRetriever(BaseRetriever):
    def __init__(self, units: List[Unit], n_process: int = 1, n_shots: int = 5, use_norm: bool = False):
        super().__init__(n_process=n_process, n_shots=n_shots)
        assert isinstance(units, list)
        assert len(units) >= n_shots

        self.units = units
        self.instance_emb = np.array([unit.source_emb for unit in units], dtype=np.float64)
        if use_norm:
            u, s, vt = np.linalg.svd(self.instance_emb, full_matrices=False)
            self.W_norm = vt.T.dot(np.diag(1 / s)).dot(vt)
            self.instance_emb = self.instance_emb.dot(self.W_norm)
        
    def _pre_compute_similarity_between_instances(self):
        x = self.instance_emb
        sim_matrix = x.dot(x.T)
        norms = np.array
    
    def _pre_compute_similarity_between_instance_target(self, targets: List[Unit]):
        if hasattr(self, "W_norm"):
            target_emb = np.array([target.source_emb.dot(self.W_norm) for target in targets])
        else:
            target_emb = np.array([target.source_emb for target in targets])


        