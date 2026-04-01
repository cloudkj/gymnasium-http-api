
import gymnasium as gym
import uuid
from app.models.schemas import EnvCreate, EnvReset, EnvAction
from app.services.manager import envs, serialize, serialize_space
from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.post("/", summary="Create an instance of the specified environment")
def create(request: EnvCreate):
    try:
        env = gym.make(request.env_id, render_mode=request.render_mode)
        instance_id = str(uuid.uuid4())
        envs[instance_id] = env
        return {"instance_id": instance_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@router.get("/", summary="List all active environment instances")
def list():
    return {id: env.spec.id for id, env in envs.items()}

@router.post("/{instance_id}/reset/", summary="Reset the environment")
def reset(instance_id: str, request: EnvReset = EnvReset()):
    if instance_id not in envs:
        raise HTTPException(status_code=404, detail="Instance not found")
    
    try:
        obs, info = envs[instance_id].reset(seed=request.seed, options=request.options)
        return {"observation": serialize(obs), "info": serialize(info)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{instance_id}/step/", summary="Step through the environment with a specified action")
def step(instance_id: str, request: EnvAction):
    if instance_id not in envs:
        raise HTTPException(status_code=404, detail="Instance not found")
    
    try:
        # Note: If the environment requires strict numpy arrays, you may need to cast req.action
        # based on envs[instance_id].action_space here. Most standard envs accept lists.
        action = request.action
        
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

@router.get("/{instance_id}/action_space/", summary="Get action space properties")
def action_space(instance_id: str):
    if instance_id not in envs:
        raise HTTPException(status_code=404, detail="Instance not found")
    return serialize_space(envs[instance_id].action_space)

@router.get("/{instance_id}/observation_space/", summary="Get observation space properties")
def observation_space(instance_id: str):
    if instance_id not in envs:
        raise HTTPException(status_code=404, detail="Instance not found")
    return serialize_space(envs[instance_id].observation_space)

@router.delete("/{instance_id}/", summary="Close and remove the environment")
def close(instance_id: str):
    if instance_id in envs:
        envs[instance_id].close()
        del envs[instance_id]
        return {"message": "Environment closed and deleted."}
    raise HTTPException(status_code=404, detail="Instance not found")
