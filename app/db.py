import asyncpg
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@db:5432/mydatabase")

async def get_db():
    try:
        return await asyncpg.connect(DATABASE_URL)
    except Exception as e:
        print("Error connecting to the database:", e)
        return None
