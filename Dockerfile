FROM python:3.10-slim

WORKDIR /app

# 1. Install system build dependencies (Fixes 'grpcio' compilation errors)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy requirements first
COPY requirements.txt .

# 3. Upgrade pip AND install wheel (Crucial for successful installs)
RUN pip install --upgrade pip setuptools wheel

# 4. Install dependencies with extended timeout (Fixes slow connection drops)
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# 5. Copy the rest of the code
COPY src/ ./src/
RUN mkdir -p models

ENTRYPOINT ["python"]