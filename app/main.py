from fastapi import FastAPI
from app.database import init_db
from app.routers.users import router as users_router
from app.routers.scenes import router as scenes_router
from app.routers.debug import router as debug_router


app = FastAPI(
    title="Interior-3D API",
    version="0.1.0",
)


@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(users_router)
app.include_router(scenes_router)
app.include_router(debug_router)
