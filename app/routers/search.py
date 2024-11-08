import asyncio
from fastapi import File, UploadFile, HTTPException
from fastapi.routing import APIRouter
from db import get_db
from index import get_minio_client, ensure_bucket_exists
from fastapi.responses import StreamingResponse
import io
search_router = APIRouter()
from classificator import classificator_instance


@search_router.post("/search/")
async def get_indexes_by_class(file: UploadFile = File(...)):
    db = await get_db()
    async with await get_minio_client() as minio_client:
        class_id = classificator_instance(file)
        try:
            images = await db.fetchrow("SELECT id FROM images WHERE class_id = $1", class_id)
            print(images)
            if images is None:
                raise HTTPException(status_code=404, detail="Image not found")
            
            # Retrieve file from MinIO
           
            return {
                "images_id": images,
            }
        except Exception as e:
            return {'problem': f'{e}' }
        finally:
            await db.close()
