from app.services.manager import envs
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image
import asyncio
import base64

router = APIRouter()

@router.get("/{instance_id}/monitor/render/", summary="Render the current state of the environment")
def render(instance_id: str):
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

async def render_frames(instance_id: str):
    env = envs[instance_id]
    while instance_id in envs:
        try:
            render_data = env.render()
            img = Image.fromarray(render_data)
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=80)
            frame = buffer.getvalue()
            
            # Yield frame in MJPEG multipart format
            yield (
                b'--frame\r\n' +
                b'Content-Type: image/jpeg\r\n\r\n' +
                frame + b'\r\n'
            )
                   
            # Throttle to roughly 30 FPS
            await asyncio.sleep(1 / 30.0)
        except Exception:
            break

@router.get("/{instance_id}/monitor/stream/", summary="Continuously stream the current state of the environment")
def stream(instance_id: str):
    if instance_id not in envs:
        raise HTTPException(status_code=404, detail="Instance not found")
        
    env = envs[instance_id]
    if env.render_mode != "rgb_array":
        raise HTTPException(
            status_code=400, 
            detail="Stream requires environment to be created with render_mode='rgb_array'."
        )
        
    # Return the StreamingResponse with the specific multipart content type
    return StreamingResponse(
        render_frames(instance_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
