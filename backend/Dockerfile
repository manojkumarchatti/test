FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV STORAGE_PATH=/app/data \
    VECTOR_STORE_PATH=/app/data/vector_store \
    UPLOADS_PATH=/app/data/uploads

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
