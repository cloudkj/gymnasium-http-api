from pydantic import BaseModel
from typing import Any, Dict, Optional

class EnvCreate(BaseModel):
    env_id: str
    render_mode: Optional[str] = None

class EnvReset(BaseModel):
    seed: Optional[int] = None
    options: Optional[Dict[str, Any]] = None

class EnvAction(BaseModel):
    action: Any
