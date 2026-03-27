import gymnasium as gym
import numpy as np
from typing import Any, Dict

# In-memory store for active environments
# Mapping: instance_id -> gym.Env
envs: Dict[str, gym.Env] = {}

def serialize(obj: Any) -> Any:
    """Recursively converts NumPy arrays and generic types to standard Python types for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]
    return obj

def get_space_info(space: gym.Space) -> Dict[str, Any]:
    """Basic serialization of Gymnasium spaces."""
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
