import gymnasium as gym
import math
import numpy as np
from typing import Any, Dict

# In-memory store for active environments
# Mapping: instance_id -> gym.Env
envs: Dict[str, gym.Env] = {}

def serialize(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return [serialize(v) for v in obj.tolist()]
    elif isinstance(obj, float):
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        if math.isnan(obj):
            return "NaN"
        return obj
    elif isinstance(obj, np.generic):
        return serialize(obj.item())
    elif isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]
    return obj

def serialize_space(space: gym.Space) -> Dict[str, Any]:
    info = {"name": space.__class__.__name__}
    if hasattr(space, "n"):
        info["n"] = int(space.n)
    if hasattr(space, "shape"):
        info["shape"] = serialize(space.shape)
    if hasattr(space, "low"):
        info["low"] = serialize(space.low)
    if hasattr(space, "high"):
        info["high"] = serialize(space.high)
    return info
