import asyncio
from fastapi import FastAPI
from routers import image_router
from model import init_db,init_classes
from routers import search_router, search_content_router
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

@app.on_event("startup")
async def startup_event():
    # Add a delay to ensure the database is ready
    await asyncio.sleep(2)
    await init_db()
    await init_classes("classes.json")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # remove this if you want to disable CORS
    allow_credentials=True,
    allow_methods=["*"],  # remove this if you want to disable CORS
    allow_headers=["*"],  # remove this if you want to disable CORS
)

app.include_router(image_router, prefix="/api/image_serv",tags=["image_serv"])
app.include_router(search_router, prefix="/api/search_serv",tags=["search_serv"])
app.include_router(search_content_router, prefix="/api/parser",tags=["parser"])