CREATE EXTENSION vector;
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
    probs FLOAT NOT NULL,
    embedding VECTOR(106) NOT NULL, 
    FOREIGN KEY (class_id) REFERENCES classes (id)
);
CREATE INDEX ON images USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON images USING hnsw (embedding vector_l1_ops);
CREATE INDEX ON images USING hnsw (embedding vector_l2_ops);