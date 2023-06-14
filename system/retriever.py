import multiprocessing as mp
import random
import numpy as np

from system.converter import Unit
from typing import List, Dict
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


class SimilarRetriever(BaseRetriever): # retriever top-k cosine similar units 
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


class DiverseRetriever(BaseRetriever): # retrieve top-k maximal marginal relevance units
    def __init__(self, units: List[Unit], n_process: int = 1, n_shots: int = 5, use_norm: bool = False, use_precompute: bool = True):
        super().__init__(n_process=n_process, n_shots=n_shots)
        assert isinstance(units, list)
        assert len(units) >= n_shots

        self.units = units
        self.instance_emb = np.array([unit.source_emb for unit in units], dtype=np.float64)
        if use_norm:
            u, s, vt = np.linalg.svd(self.instance_emb, full_matrices=False)
            self.W_norm = vt.T.dot(np.diag(1 / s)).dot(vt)
            self.instance_emb = self.instance_emb.dot(self.W_norm)
        self.use_precompute = use_precompute
        
    def compute_similarity_between_instances(self, use_precompute):
        x = self.instance_emb
        sim_matrix = x.dot(x.T)
        norms = np.array([np.array([np.sqrt(np.diagonal(sim_matrix))])])
        sim_matrix = sim_matrix / norms / norms.T
        if not use_precompute:
            return sim_matrix
        self.sim_instance_memory = mp.shared_memory.SharedMemory(create=True, size=sim_matrix.nbytes, name="similarity_between_instances")
        buffer = np.ndarray(sim_matrix.shape, dtype=sim_matrix.dtype, buffer=self.sim_instance_memory.buf)
        buffer[:] = sim_matrix

    def compute_similarity_between_instance_target(self, targets: List[Unit], use_precompute): 
        x = self.instance_emb
        if hasattr(self, "W_norm"):
            y = np.array([target.source_emb.dot(self.W_norm) for target in targets])
        else:
            y = np.array([target.source_emb for target in targets])
        norms = np.sqrt((y ** 2).sum(axis=1)).reshape(-1, 1)
        sim_matrix = (y.dot(x.T) / norms)
        if not use_precompute:
            return sim_matrix
        self.sim_target_memory = mp.shared_memory.SharedMemory(create=True, size=sim_matrix.nbytes, name="similarity_between_instance_target")
        buffer = np.ndarray(sim_matrix.shape, dtype=sim_matrix.dtype, buffer=self.sim_target_memory.buf)
        buffer[:] = sim_matrix
    
    def build_memory(self, targets: List[Unit]): # please remember to close or unlink
        self.compute_similarity_between_instances(use_precompute=True)
        self.compute_similarity_between_instance_target(targets=targets, use_precompute=True)
    
    def release_memory(self): # close or unlink memory
        if hasattr(self, "sim_instance_memory"):
            self.sim_instance_memory.close()
            self.sim_instance_memory.unlink()
        if hasattr(self, "sim_target_memory"):
            self.sim_target_memory.close()
            self.sim_target_memory.unlink()
    
    def get_samples(self, target_idx: int, shared_dict: Dict) -> List[Unit]:
        assert shared_dict
        sim_instance = shared_dict["similarity_between_instances"]
        sim_target = shared_dict["similarity_between_instance_target"]
        
        sim_target = sim_target[target_idx]
        
        sample_ids = []
        while len(sample_ids) < self.n_shots:
            pool = [idx for idx in range(len(sim_instance)) if idx not in sample_ids]
            scores = []

            for i in pool:
                if sample_ids:
                    max_score = max(sim_instance[i][j] for j in sample_ids)
                else:
                    max_score = 0
                score = sim_target[i] - max_score
                scores.append(score)
            
            sample_ids.append(pool[np.argmax(scores)])

        samples = [self.units[idx] for idx in sample_ids]
        return samples
    
    def get_batch_samples(self, target_indices: List[int], targets: List[Unit] = None) -> List[List[Unit]]: # todo: opt no pre-compute version
        if self.n_process <= 1:
            return [self.get_samples(target_idx) for target_idx in target_indices]
        
        with mp.Manager() as manager:
            shared_dict = manager.dict()
            if self.use_precompute:
                assert hasattr(self, "sim_instance_memory") and hasattr(self, "sim_target_memory"), "run build_memory and release_memory"
                buffer = self.sim_instance_memory.buf
                shared_dict["similarity_between_instances"] = buffer[:]
                buffer = self.sim_target_memory.buf
                shared_dict["similarity_between_instance_target"] = buffer[:]
            else:
                assert targets
                shared_dict["similarity_between_instances"] = self.compute_similarity_between_instances(use_precompute=False)
                shared_dict["similarity_between_instance_target"] = self.compute_similarity_between_instance_target(use_precompute=False)
            
            with mp.Pool(self.n_process) as pool:
                batch_samples = pool.starmap(self.get_samples, [(idx, shared_dict) for idx in target_indices])
            return batch_samples
