# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy poetry configuration files first for dependency installation
COPY pyproject.toml poetry.lock /app/

# Install system dependencies needed for Playwright and the app
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    wget \
    gnupg \
    libnss3 \
    libatk1.0 \
    libatk-bridge2.0 \
    libcups2 \
    libxcomposite1 \
    libxrandr2 \
    libxdamage1 \
    libxkbcommon0 \
    libgbm1 \
    libpango-1.0-0 \
    libasound2 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Ensure Poetry is in PATH
ENV PATH="/root/.local/bin:$PATH"

# Install dependencies from poetry
RUN poetry config virtualenvs.create false \
 && poetry install --no-root --no-interaction --no-ansi

# Install Playwright and its browsers
RUN pip install playwright \
 && playwright install --with-deps
COPY download_script.py .

 # Set up the default command to run your Python script
RUN ["python", "download_script.py"]
 
# Copy the app code after installing dependencies to optimize caching
COPY . .

# Expose port 8000
EXPOSE 8000

# Run uvicorn server with FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
