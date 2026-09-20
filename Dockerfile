FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces expects the app to listen on port 7860.
ENV DATA_DIR=/code/data
RUN mkdir -p /code/data
EXPOSE 7860

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]