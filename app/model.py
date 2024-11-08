
import asyncpg
from db import DATABASE_URL
async def init_db():
    #add enum class 
    conn = await asyncpg.connect(DATABASE_URL)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            minio_path TEXT NOT NULL,
            content_type TEXT NOT NULL,
            class_id INTEGER NOT NULL
        );
    """)
    await conn.close()
