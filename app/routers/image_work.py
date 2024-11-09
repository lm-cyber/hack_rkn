import asyncio
from fastapi import File, UploadFile, HTTPException
from fastapi.routing import APIRouter
from db import get_db
from index import get_minio_client, ensure_bucket_exists
from fastapi.responses import StreamingResponse
import io
import numpy as np
image_router = APIRouter()
import random
from classificator import classificator_instance
from typing import Union
@image_router.post("/images/")
async def upload_image(file: UploadFile = File(...),page_url: Union[str, None] = None):
    db = await get_db()
    async with await get_minio_client() as minio_client:
        await ensure_bucket_exists(minio_client)

        minio_path = f"{file.filename}"

        # Upload to MinIO
        try:
            await minio_client.put_object(
                Bucket="images",
                Key=minio_path,
                Body=file.file,
                ContentType=file.content_type
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"MinIO upload error: {e}")

        # Store metadata in PostgreSQL
        try:
            result:dict= classificator_instance.predict_result(file)# model class TODO!!!!!!!!!!!!!!!!!!!!!!!!!!
           
            image_id = await db.fetchval(
                """
                INSERT INTO images (
                    filename,
                    minio_path,
                    content_type,
                    class_id,
                    page_url,
                    embedding,
                    probs
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
                """,
                file.filename,
                minio_path,
                file.content_type,
                result['class'],# model class TODO!!!!!!!!!!!!!!!!!!!!!!!!!!
                page_url,
                str(result['embedding']),
                result['probs_class']
                
            )
 
        finally:
            await db.close()
          
     
        return {
            "id": image_id,
            "filename": file.filename,
            "path": minio_path,
            **result
            
            
            }

@image_router.get("/images_source/{image_id}")
async def get_image_source(image_id: int):
    db = await get_db()
    async with await get_minio_client() as minio_client:

        try:
            image = await db.fetchrow("SELECT * FROM images WHERE id = $1", image_id)
            if image is None:
                raise HTTPException(status_code=404, detail="Image not found")
            
            # Retrieve file from MinIO
            response = await minio_client.get_object(Bucket="images", Key=image['minio_path'])
            content = await response['Body'].read()
            return StreamingResponse(io.BytesIO(content), media_type=image['content_type'])

        finally:
            await db.close()
@image_router.get("/images_metadata/{image_id}")
async def get_image_metadata(image_id: int):
    db = await get_db()
    async with await get_minio_client() as minio_client:

        try:
            image = await db.fetchrow("SELECT * FROM images WHERE id = $1", image_id)
            if image is None:
                raise HTTPException(status_code=404, detail="Image not found")
            
            return image

        finally:
            await db.close()

@image_router.delete("/images/{image_id}")
async def delete_image(image_id: int):
    db = await get_db()
    async with await get_minio_client() as minio_client:

        try:
            image = await db.fetchrow("SELECT * FROM images WHERE id = $1", image_id)
            if image is None:
                raise HTTPException(status_code=404, detail="Image not found")

            # Delete from MinIO
            await minio_client.delete_object(Bucket="images", Key=image['minio_path'])

            # Delete from PostgreSQL
            await db.execute("DELETE FROM images WHERE id = $1", image_id)
        finally:
            await db.close()

        return {"detail": "Image deleted"}
