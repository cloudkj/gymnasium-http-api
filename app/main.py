from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import envs, monitor

app = FastAPI(title="Gymnasium HTTP API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(envs.router, prefix="/v1/envs", tags=["Environments"])
app.include_router(monitor.router, prefix="/v1/envs", tags=["Monitoring"])
app.mount("/", StaticFiles(directory="app/static"), name="static")
