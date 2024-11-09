
import asyncpg
from db import DATABASE_URL
import json
async def init_db():
    #add enum class 
    conn = await asyncpg.connect(DATABASE_URL)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS classes (
            id SERIAL PRIMARY KEY,
            "name" TEXT UNIQUE
        );

        CREATE TABLE IF NOT EXISTS images (
            id SERIAL PRIMARY KEY,
            "filename" TEXT NOT NULL,
            minio_path TEXT NOT NULL,
            content_type TEXT NOT NULL,
            class_id INTEGER NOT NULL,
            page_url TEXT,
            predict_prob FLOAT NOT NULL,
            embedding VECTOR(3) NOT NULL, 
            probs VECTOR(3) NOT NULL,
            FOREIGN KEY (class_id) REFERENCES classes (id)
        );

    """)
    await conn.close()


async def init_classes(path_classes_map):
    conn = await asyncpg.connect(DATABASE_URL)

    with open(path_classes_map) as f:
        classes = json.load(f)

    for k, v in sorted(classes.items(), key=lambda item: item[0]):
        await conn.execute(
                "INSERT INTO classes (name) VALUES ($1) ON CONFLICT (name) DO NOTHING",
                v
            )
    conn.close()
   
    await conn.close()
