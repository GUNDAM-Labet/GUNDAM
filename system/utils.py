from typing import Dict
import pandas as pd
from torch.utils.data import Dataset
from dataclasses import dataclass
import numpy as np
from typing import List, Optional, Iterator


@dataclass
class Unit:
    unit_id: str
    source_input: str
    target_output: str
    source_emb: Optional[np.ndarray] = None
    target_emb: Optional[np.ndarray] = None
    priority_level: int


