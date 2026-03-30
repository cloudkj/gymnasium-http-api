from app.services.manager import envs
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image
import asyncio

router = APIRouter()

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
