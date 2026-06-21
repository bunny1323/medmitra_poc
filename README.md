---
title: MedMitra RAG API
emoji: ⚕️
sdk: docker
app_port: 7860
---

# MedMitra RAG API

FastAPI backend for MedMitra semantic search and RAG.
MedMitra is a healthcare-information search and RAG backend built with **FastAPI**, **Qdrant (Hybrid Search)**, **PyMuPDF**, and **Groq (Llama)**. 

This API provides secure endpoints to ingest multiple medical textbooks (PDFs), runs deterministic emergency and symptom severity classifications, retrieves context using a hybrid vector-search schema, and generates safe, grounded information responses.

---

## 🏗 Architecture

```text
               Patient Query
                     │
                     ▼
       [Deterministic Emergency Check] ──(Emergency)──► [Direct 108 Bypass Response]
                     │
               (Non-Emergency)
                     ▼
           [Secure API Header Key]
                     │
                     ▼
       [Hybrid Search Retrieval (Qdrant)]
         ├── Dense Embeddings: BAAI/bge-small-en-v1.5
         └── Sparse Embeddings: prithivida/Splade_PP_en_v1
                     │
                     ▼
         [Reciprocal Rank Fusion (RRF)]
                     │
                     ▼
       [Deterministic Severity Service] ──► Normal / Urgent / Critical
                     │
                     ▼
          [Safe LLM Prompt (Groq)] ─────► Empathetic Answer & Cautions
                     │
                     ▼
               JSON Response
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.11+
- Docker (optional, for containerized deployments)

### 2. Install Dependencies
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 3. Add Medical Guidelines/Books
Place your medical textbook PDF files (e.g. `icmr_stw_volume_1.pdf`, `who_imci_chart_booklet.pdf`) under the `data/books/` directory:
```text
data/books/icmr_stw_volume_1.pdf
data/books/icmr_stw_volume_3.pdf
data/books/who_imci_chart_booklet.pdf
```

### 4. Configure Environment Variables
Copy the `.env.example` template:
```powershell
copy .env.example .env
```
Edit `.env` and fill in:
- `INTERNAL_API_KEY`: The API key protecting your backend endpoints (sent via `X-Internal-API-Key` header).
- `GROQ_API_KEY`: Your Groq platform access key.
- `QDRANT_URL` and `QDRANT_API_KEY` (Leave blank for local embedded Qdrant database fallback).

---

## 📚 Multi-Book Ingestion Pipeline

MedMitra features a flexible, multi-book ingestion engine that parses page text, generates dense & sparse vectors, skips duplicates using SHA-256 hashes, and tracks status inside `data/registry/documents.json`.

Run ingestion from the command line:

```powershell
# Ingest any new books (Scan data/books/ and skip duplicates)
python scripts/ingest_books.py --mode append

# Full rebuild (Drops the collection and re-indexes all files)
python scripts/ingest_books.py --mode rebuild

# Replace a single book by its registry ID (re-ingests the file)
python scripts/ingest_books.py --mode replace --id <source_uuid>

# Delete a single book by its registry ID
python scripts/ingest_books.py --mode delete --id <source_uuid>
```

---

## 🧪 Running Retrieval Evaluation

A retrieval evaluation pipeline measures the search quality baseline over 30 medical questions.
To run the latency and MRR evaluation:
```powershell
python scripts/evaluate_retrieval.py
```

---

## 🚀 Running the Server

Start the local development server:
```powershell
uvicorn app.main:app --reload
```
The application will start at `http://127.0.0.1:8000`.

---

## 🐳 Docker Deployment

Build and run the backend locally using Docker:
```bash
docker build -t medmitra-backend .
docker run -p 7860:7860 --env-file .env medmitra-backend
```

### Hugging Face Docker Spaces Deployment
To deploy this API on Hugging Face Spaces:
1. Create a new Space on Hugging Face, select **Docker** as the SDK.
2. Select the **Blank** template.
3. Commit this repository's codebase (including the `Dockerfile` and `requirements.txt`).
4. In the Space **Settings**, add your environment secrets:
   - `INTERNAL_API_KEY`
   - `GROQ_API_KEY`
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
5. The container will automatically build and expose port `7860`.

---

## 🔌 API Endpoints Documentation

### Health Checks (Public)
- **GET** `/health/live` - Verify if the API server is alive.
- **GET** `/health/ready` - Verify if Groq is configured, Qdrant is connected, and points are loaded.

### Queries & Admin (Protected with `X-Internal-API-Key` header)

#### 1. Query Endpoint
- **URL**: `/api/v1/query`
- **Method**: `POST`
- **Headers**: 
  - `Content-Type: application/json`
  - `X-Internal-API-Key: <your_key_here>`
- **Request Body**:
  ```json
  {
    "query": "I have fever and cough",
    "top_k": 5
  }
  ```
- **Response Structure**:
  ```json
  {
    "query": "I have fever and cough",
    "answer_mode": "retrieval_grounded",
    "severity_index": "NORMAL",
    "severity_reasons": [],
    "retrieval_relevance_score": 0.74,
    "retrieval_relevance_level": "MEDIUM",
    "confidence_note": "This score represents retrieval relevance only. It is not a diagnosis probability.",
    "answer": "The medical guidelines recommend rest and monitoring temperature...",
    "home_cautions": ["Rest", "Hydrate"],
    "sources": [],
    "error": null
  }
  ```

#### 2. Get Registered Books
- **URL**: `/api/v1/admin/books`
- **Method**: `GET`

#### 3. Trigger Reindexing Pipeline
- **URL**: `/api/v1/admin/reindex`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "mode": "append|rebuild|replace|delete",
    "source_id": "<uuid_of_book_for_replace_or_delete>"
  }
  ```

#### 4. Parse Prescription Endpoint
- **URL**: `/api/v1/prescription/parse`
- **Method**: `POST`
- **Headers**: `Content-Type: multipart/form-data`
- **Body**: Upload an image file under the key `file`.

**cURL Example**:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/prescription/parse" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/prescription_image.jpg"
```

---

## 🔮 Future Benchmarking & Evaluation
Current embedding models are kept stable as:
- **Dense model**: `BAAI/bge-small-en-v1.5`
- **Sparse model**: `prithivida/Splade_PP_en_v1`

As a future roadmap item, the following specialized clinical retrieval models will be benchmarked later:
- **MedCPT Query Encoder + MedCPT Article Encoder + BM25 + RRF**
