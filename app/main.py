import asyncio
from fastapi import FastAPI
from routers import image_router
from model import init_db
from routers import search_router
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # Add a delay to ensure the database is ready
    await asyncio.sleep(2)
    await init_db()


app.include_router(image_router, prefix="/api")
app.include_router(search_router, prefix="/api")