import numpy as np

from typing import Optional
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