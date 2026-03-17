FROM python:3.10-slim
WORKDIR /app

# Install all necessary libraries
RUN pip install --no-cache-dir fastapi uvicorn torch httpx pydantic requests numpy

COPY . /app

# The command to run depends on the folder it is in
CMD if [ -f "app.py" ] && grep -q "FastAPI" app.py; then \
        uvicorn app:app --host 0.0.0.0 --port 8000; \
    else \
        python app.py; \
    fi