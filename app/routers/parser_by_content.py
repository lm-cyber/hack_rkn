import asyncio
import asyncio
from fastapi import File, UploadFile, HTTPException
from fastapi.routing import APIRouter
from db import get_db
from index import get_minio_client, ensure_bucket_exists
from fastapi.responses import StreamingResponse
import io

search_content_router = APIRouter()


@search_content_router.post("/pars/")
async def get_indexes_by_class(file: UploadFile = File(...)):
    return {
        "text":"dfs search list txt sytes"
    }
