import uuid
from typing import Any, Dict, Optional
import base64
from io import BytesIO
from PIL import Image

import gymnasium as gym
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Gymnasium HTTP API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins (fine for local development)
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, DELETE, etc.)
    allow_headers=["*"],
)

# In-memory store for active environments
# Mapping: instance_id -> gym.Env
envs: Dict[str, gym.Env] = {}

# --- Pydantic Models for Requests ---

class EnvCreate(BaseModel):
    env_id: str
    render_mode: Optional[str] = None

class EnvReset(BaseModel):
    seed: Optional[int] = None
    options: Optional[Dict[str, Any]] = None

class EnvAction(BaseModel):
    action: Any  # Can be int, list of floats, dict, etc., depending on the action space

# --- Helper Functions ---

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

# --- API Endpoints ---

@app.post("/v1/envs/", summary="Create an instance of the specified environment")
def create_env(req: EnvCreate):
    try:
        env = gym.make(req.env_id, render_mode=req.render_mode)
        instance_id = str(uuid.uuid4())
        envs[instance_id] = env
        return {"instance_id": instance_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/v1/envs/", summary="List all active environment instances")
def list_envs():
    return {"all_env_ids": {iid: env.spec.id for iid, env in envs.items()}}

@app.post("/v1/envs/{instance_id}/reset/", summary="Reset the environment")
def reset_env(instance_id: str, req: EnvReset = EnvReset()):
    if instance_id not in envs:
        raise HTTPException(status_code=404, detail="Instance not found")
    
    try:
        obs, info = envs[instance_id].reset(seed=req.seed, options=req.options)
        return {"observation": serialize(obs), "info": serialize(info)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/envs/{instance_id}/step/", summary="Step the environment")
def step_env(instance_id: str, req: EnvAction):
    if instance_id not in envs:
        raise HTTPException(status_code=404, detail="Instance not found")
    
    try:
        # Note: If the environment requires strict numpy arrays, you may need to cast req.action
        # based on envs[instance_id].action_space here. Most standard envs accept lists.
        action = req.action
        
        obs, reward, terminated, truncated, info = envs[instance_id].step(action)
        return {
            "observation": serialize(obs),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": serialize(info)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/envs/{instance_id}/action_space/", summary="Get action space properties")
def get_action_space(instance_id: str):
    if instance_id not in envs:
        raise HTTPException(status_code=404, detail="Instance not found")
    return {"info": get_space_info(envs[instance_id].action_space)}

@app.get("/v1/envs/{instance_id}/observation_space/", summary="Get observation space properties")
def get_observation_space(instance_id: str):
    if instance_id not in envs:
        raise HTTPException(status_code=404, detail="Instance not found")
    return {"info": get_space_info(envs[instance_id].observation_space)}

@app.get("/v1/envs/{instance_id}/render/", summary="Render the current state of the environment")
def render_env(instance_id: str):
    if instance_id not in envs:
        raise HTTPException(status_code=404, detail="Instance not found")
    
    env = envs[instance_id]
    
    if env.render_mode is None or env.render_mode == "human":
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot render. Environment was created with render_mode='{env.render_mode}'. "
                   "Must be created with a non-human mode like 'rgb_array' or 'ansi'."
        )
        
    try:
        render_data = env.render()
        
        # Handle RGB Array by converting to a base64 encoded PNG
        if env.render_mode == "rgb_array":
            # Convert numpy array to PIL Image
            img = Image.fromarray(render_data)
            
            # Save image to an in-memory buffer as a PNG
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            
            # Encode the buffer bytes to base64 string
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            return {
                "render_mode": env.render_mode,
                "content_type": "image/png",
                "encoding": "base64",
                "data": img_base64
            }
            
        # Handle list of RGB arrays (e.g., if recording video frames)
        elif env.render_mode == "rgb_array_list":
            frames_base64 = []
            for frame in render_data:
                img = Image.fromarray(frame)
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                frames_base64.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
            
            return {
                "render_mode": env.render_mode,
                "content_type": "image/png",
                "encoding": "base64",
                "data": frames_base64
            }
            
        # Fallback for other modes like "ansi" (returns a string)
        else:
            return {
                "render_mode": env.render_mode,
                "content_type": "text/plain" if isinstance(render_data, str) else "application/json",
                "encoding": "utf-8",
                "data": serialize(render_data)
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/v1/envs/{instance_id}/", summary="Close and remove the environment")
def close_env(instance_id: str):
    if instance_id in envs:
        envs[instance_id].close()
        del envs[instance_id]
        return {"message": "Environment closed and deleted."}
    raise HTTPException(status_code=404, detail="Instance not found")

app.mount("/", StaticFiles(directory="app/static"), name="static")
