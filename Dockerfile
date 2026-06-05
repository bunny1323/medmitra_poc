FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential libglib2.0-0 libsm6 libxrender1 libxext6     && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
RUN groupadd -r medmitra && useradd -r -g medmitra -d /workspace medmitra
COPY app/ ./app/
COPY data/ ./data/
COPY docs/ ./docs/
COPY scripts/ ./scripts/
RUN mkdir -p data/books data/reports && chown -R medmitra:medmitra /workspace
EXPOSE 8000
USER medmitra
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
