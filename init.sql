CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    minio_path TEXT NOT NULL,
    content_type TEXT NOT NULL
    -- class_ Enum ('image/jpeg', 'image/png'),
    -- PAGE_URL URL,
);
