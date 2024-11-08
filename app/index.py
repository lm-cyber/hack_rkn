import os
from aiobotocore.session import get_session
from botocore.exceptions import ClientError

MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "images")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")


async def get_minio_client():
    session = get_session()
    client =  session.create_client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1",
    )
    return client


async def ensure_bucket_exists(client):
    try:
        response = await client.list_buckets()
        bucket_names = [bucket["Name"] for bucket in response["Buckets"]]
        
        if MINIO_BUCKET_NAME not in bucket_names:
            await client.create_bucket(Bucket=MINIO_BUCKET_NAME)
            print(f"Bucket {MINIO_BUCKET_NAME} created.")
        else:
            print(f"Bucket {MINIO_BUCKET_NAME} already exists.")
    except ClientError as e:
        print(f"An error occurred: {e}")
