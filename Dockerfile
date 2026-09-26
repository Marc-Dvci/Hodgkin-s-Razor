# CPU image: runs the demo, the web application, the tests and the report.
# Simulation and training need a CUDA device; see README for the GPU image.
FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .
EXPOSE 8000
CMD ["python", "app/server.py", "--no-gpu", "--host", "0.0.0.0"]
