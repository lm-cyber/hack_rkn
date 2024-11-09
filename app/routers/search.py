import asyncio
from fastapi import File, UploadFile, HTTPException
from fastapi.routing import APIRouter
from db import get_db
from index import get_minio_client, ensure_bucket_exists
from fastapi.responses import StreamingResponse
import io
search_router = APIRouter()
from classificator import classificator_instance
from typing import Literal

@search_router.post("/search/")
async def get_indexes_by_class(
    file: UploadFile = File(...),
    search_by:Literal["class","one_shot_embedding"] = "class",
    distance_type:Literal["cosine","euclidean","manhattan"] = "cosine"
    ):
    """
    search_by: "class","one_shot_embedding"
    distance_type: "cosine","euclidean","manhattan" distance not in class search
    """
    db = await get_db()
    async with await get_minio_client() as minio_client:
        try:
            if distance_type == "cosine":
                operator = '<=>'
            elif distance_type == "euclidean":
                operator = '<->'
            else:
                operator = '<+>'

            if search_by == "class":
                class_id = classificator_instance(file)
                images = await db.fetch("SELECT id , probs FROM images WHERE class_id = $1 order by 2 desc" , class_id)
            elif search_by == "one_shot_embedding":
                
                embedding = classificator_instance.predict_embedding(file)
                images = await db.fetch(
                    f"""SELECT id, 1 - (embedding {operator} '""" + str(embedding) + """'::vector(3)) AS cosine_similarity_embs FROM images order by 2 desc""")
            if images is None:
                raise HTTPException(status_code=404, detail="Image not found")
            
            # Retrieve file from MinIO
           
            return [dict(row) for row in images]

        finally:
            await db.close()


@search_router.get("/search_class/")
async def get_indexes_by_class(
      class_name: str
    ):
   
    db = await get_db()
    result = await db.fetch("""
    SELECT images.id, images.probs
    FROM images
    JOIN classes ON images.class_id = classes.id
    WHERE classes.name = $1
    ORDER BY images.probs DESC;
    """ , class_name)

    return [dict(row) for row in result]