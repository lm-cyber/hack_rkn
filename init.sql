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
    embedding VECTOR(3) NOT NULL, 
    predict_prob VECTOR(3) NOT NULL,
    FOREIGN KEY (class_id) REFERENCES classes (id)
);
