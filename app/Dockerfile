FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml poetry.lock /app/

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
 && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
RUN poetry config virtualenvs.create false \
 && poetry install --no-root --no-interaction --no-ansi

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
