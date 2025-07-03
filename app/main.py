from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routers.users import router as users_router
from app.routers.scenes import router as scenes_router
from app.routers.debug import router as debug_router

app = FastAPI(
    title="Interior-3D API",
    version="0.1.0",
)

# Allow requests coming from your Next.js front-end origin(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",          # if you ever browse locally
        "http://10.220.202.24:3000"       # your EC2‐in‐VPC front-end URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(users_router)
app.include_router(scenes_router)
app.include_router(debug_router)
