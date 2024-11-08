import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException
from db import get_db
from index import get_minio_client, ensure_bucket_exists
from model import init_db
from fastapi.responses import StreamingResponse
import io
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # Add a delay to ensure the database is ready
    await asyncio.sleep(2)
    await init_db()

@app.post("/images/")
async def upload_image(file: UploadFile = File(...)):
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
            image_id = await db.fetchval(
                """
                INSERT INTO images (filename, minio_path, content_type)
                VALUES ($1, $2, $3)
                RETURNING id
                """,
                file.filename,
                minio_path,
                file.content_type
            )
        finally:
            await db.close()

        return {"id": image_id, "filename": file.filename, "path": minio_path}

@app.get("/images/{image_id}")
async def get_image(image_id: int):
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

@app.put("/images/{image_id}")
async def update_image(image_id: int, file: UploadFile = File(...)):
    db = await get_db()
    async with await get_minio_client() as minio_client:

        try:
            image = await db.fetchrow("SELECT * FROM images WHERE id = $1", image_id)
            if image is None:
                raise HTTPException(status_code=404, detail="Image not found")

            minio_path = f"{file.filename}"
            
            # Update MinIO
            await minio_client.put_object(
                Bucket="images",
                Key=minio_path,
                Body=file.file,
                ContentType=file.content_type
            )

            # Update PostgreSQL metadata
            await db.execute(
                """
                UPDATE images SET filename = $1, minio_path = $2, content_type = $3 WHERE id = $4
                """,
                file.filename,
                minio_path,
                file.content_type,
                image_id
            )
        finally:
            await db.close()

        return {"filename": file.filename, "path": minio_path}

@app.delete("/images/{image_id}")
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
