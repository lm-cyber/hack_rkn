import asyncio
from fastapi import File, UploadFile, HTTPException
from fastapi.routing import APIRouter
from db import get_db
from index import get_minio_client, ensure_bucket_exists
from fastapi.responses import StreamingResponse
import io
import random
from classificator import classificator_instance
from typing import Union
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from fastapi.routing import APIRouter
import re
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from PIL import Image
import io

def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)
    return sanitized

import uuid

def generate_unique_filename(filename: str) -> str:
    extension = filename.split('.')[-1]
    return f"{uuid.uuid4().hex}.{extension}"
search_content_router = APIRouter()

async def fetch_image_as_file(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                content_type = response.headers.get('Content-Type', '')  
                image_bytes = await response.read()
                return io.BytesIO(image_bytes), content_type  
            else:
                raise HTTPException(status_code=404, detail="Image not found")

async def upload_image(file: UploadFile, content_type: str, page_url: Union[str, None] = None):
    db = await get_db()
    async with await get_minio_client() as minio_client:
        await ensure_bucket_exists(minio_client)

        minio_path = f"{file.filename}"
        minio_path = sanitize_filename(minio_path)
        minio_path =generate_unique_filename(minio_path)
        # Upload to MinIO
        try:
            await minio_client.put_object(
                Bucket="images",
                Key=minio_path,
                Body=file.file,
                ContentType=content_type 
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"MinIO upload error: {e}")


        try:
            im = Image.open(file.file)
            im = im.convert("RGB")
            result: dict = classificator_instance.predict_result(im)  
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
                content_type,  # Store content type in DB
                result['class'],  # model class TODO
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
            "class" :result['class']
        }




async def fetch_images(url: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)  # Headless mode
        page = await browser.new_page()
        
        await page.goto(url)
        await page.wait_for_load_state('networkidle')  # Wait for network activity to idle

        # Get page content after JavaScript execution
        html = await page.content()
        soup = BeautifulSoup(html, 'html.parser')

        # Example: Extract all image URLs
        img_tags = soup.find_all('img')
        img_urls = [img['src'] for img in img_tags if img.get('src')]

        await browser.close()
        return img_urls



@search_content_router.post("/pars/")
async def get_indexes_by_class(text: str):
    all_imgs = dict()
    for url in text.split("\n"):
        all_imgs[url] = await fetch_images(url)
    reuslt = []

    for url, imgs in all_imgs.items():
        for img in imgs:
            try:
                # Fetch the image as a file-like object and the content type
                image_file, content_type = await fetch_image_as_file(img)
                
                # Create an UploadFile instance from the image bytes
                upload_file = UploadFile(filename=img, file=image_file)
                
                # Upload the image with the correct content type
                reuslt.append(await upload_image(upload_file, content_type, url))
            except Exception as e:
                # Handle any error while fetching or uploading the image
                print(f"Error processing image {img} from {url}: {e}")

    return reuslt



'''
comand 


dir/[main.hash.js css.hash.css hstml.html]


'''