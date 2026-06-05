# MedMitra ML Search Service

A medical-information retrieval service using **MedCPT** dense embeddings, **BM25** sparse retrieval and **Reciprocal Rank Fusion** over a **Qdrant** vector store. Data sourced from Kaggle datasets.

> ⚠️ **Prototype Notice**: All Kaggle-derived records carry `review_status = "prototype_unverified"`. This service is not a clinical tool. It does not diagnose diseases or prescribe medicines.

---

## Features

| Feature | Status |
|---|---|
| Disease / symptom information search | ✅ |
| Medicine information search with typo correction | ✅ |
| Hybrid MedCPT + BM25 retrieval with RRF fusion | ✅ |
| Qdrant vector store (dense + sparse named vectors) | ✅ |
| Emergency keyword detection (deterministic rules) | ✅ |
| Restricted agent with tool routing | ✅ |
| Grounded LLM responses (Groq) | ✅ |
| Antibiotic prescription warnings | ✅ |
| API key authentication (X-Internal-API-Key) | ✅ |
| Data cleaning pipeline for Kaggle datasets | ✅ |
| Rate limiting & CORS middleware | ✅ |

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Docker (for Qdrant)
- Kaggle account (for dataset download)

### 2. Clone and install

```bash
cd medmitra_ml_service
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Linux / Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment variables

```bash
cp .env.example .env
# Edit .env — set INTERNAL_API_KEY, GROQ_API_KEY (optional), QDRANT_API_KEY (cloud only)
```

Generate a secure API key:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 4. Start Qdrant

**Linux / Mac:**
```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
  qdrant/qdrant
```

**Windows PowerShell:**
```powershell
docker run -p 6333:6333 -p 6334:6334 `
  -v "${PWD}/qdrant_storage:/qdrant/storage:z" `
  qdrant/qdrant
```

---

## Kaggle Dataset Setup

### Download Commands

Set up Kaggle credentials first:
```bash
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or %USERPROFILE%\.kaggle\ (Windows)
# Get your API token at: https://www.kaggle.com/settings → API
```

Download datasets into `app/data/raw/`:
```bash
kaggle datasets download -d itachi9604/disease-symptom-description-dataset -p app/data/raw --unzip
kaggle datasets download -d niyarrbarman/symptom2disease -p app/data/raw --unzip
kaggle datasets download -d palakjain9/1000-drugs-and-side-effects -p app/data/raw --unzip
kaggle datasets download -d mohneesh7/indian-medicine-data -p app/data/raw --unzip
```

**Windows PowerShell:**
```powershell
kaggle datasets download -d itachi9604/disease-symptom-description-dataset -p app\data\raw --unzip
kaggle datasets download -d niyarrbarman/symptom2disease -p app\data\raw --unzip
kaggle datasets download -d palakjain9/1000-drugs-and-side-effects -p app\data\raw --unzip
kaggle datasets download -d mohneesh7/indian-medicine-data -p app\data\raw --unzip
```

Rename CSV files if needed to match expected names:
- `disease_symptom_description_dataset.csv`
- `symptom2disease.csv`
- `1000_drugs_and_side_effects.csv`
- `indian_medicine_data.csv`

---

## Data Pipeline

### Step 1: Clean datasets

```bash
python -m app.scripts.clean_datasets
```

Outputs:
- `app/data/processed/diseases.json` — normalized disease records
- `app/data/processed/medicines.json` — merged medicine records
- `app/data/processed/symptom_queries.json` — evaluation queries

### Step 2: Build Qdrant index

```bash
python -m app.scripts.build_qdrant_index
```

This embeds all records with:
- **Dense**: `ncbi/MedCPT-Article-Encoder` (cosine similarity)
- **Sparse**: `fastembed Qdrant/bm25` (BM25 sparse vectors)

And upserts everything into the `medmitra_knowledge` collection.

---

## Run the Service

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI: http://localhost:8000/docs

---

## API Endpoints

All non-health endpoints require the header:
```
X-Internal-API-Key: <your-key>
```

### Health
| Method | Path | Auth |
|--------|------|------|
| GET | `/health/live` | None |
| GET | `/health/ready` | None |
| GET | `/metrics` | None |

### Search
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/search/disease` | Disease / symptom information search |
| POST | `/api/v1/search/medicine` | Medicine information search |
| POST | `/api/v1/emergency-check` | Deterministic emergency keyword check |

### Agent & RAG
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/agent/chat` | Restricted tool-routing medical information agent |
| POST | `/api/v1/chat/rag` | Hybrid retrieval RAG (book-based legacy) |

### Admin
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/admin/clean-data` | Run data cleaning pipeline |
| POST | `/api/v1/admin/rebuild-index` | Rebuild Qdrant index |
| GET | `/api/v1/admin/index-status` | Index and infrastructure status |

---

## Swagger Testing Steps

1. Open http://localhost:8000/docs
2. Click **Authorize** → enter your `INTERNAL_API_KEY`
3. Test the following sample requests:

### Disease search
```json
POST /api/v1/search/disease
{
  "query": "i have fever cold and mild cough from yesterday",
  "age_group": "adult",
  "duration_days": 1,
  "top_k": 5
}
```

### Medicine search (with typo)
```json
POST /api/v1/search/medicine
{
  "query": "paracetmol uses",
  "top_k": 5,
  "allow_typo_correction": true
}
```

### Emergency check
```json
POST /api/v1/emergency-check
{
  "text": "difficulty breathing and severe chest pain"
}
```

### Agent chat
```json
POST /api/v1/agent/chat
{
  "query": "what are the uses of azithromycin",
  "session_id": "test-001"
}
```

---

## Postman Request Bodies

### Disease search
```json
{
  "query": "fever cold and mild cough",
  "age_group": "adult",
  "top_k": 5,
  "include_full_text": false
}
```

### Medicine search
```json
{
  "query": "paracetmol uses",
  "top_k": 5,
  "allow_typo_correction": true
}
```

### Emergency check
```json
{
  "text": "difficulty breathing and severe chest pain"
}
```

### Agent chat
```json
{
  "query": "runny nose sneezing what could it be",
  "session_id": "session-abc"
}
```

---

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/unit/ -v

# Specific group
python -m pytest tests/unit/test_search_features.py::TestEmergencyDetection -v
```

---

## Environment Variables Reference

| Variable | Description | Required |
|---|---|---|
| `INTERNAL_API_KEY` | API authentication key | ✅ |
| `QDRANT_URL` | Qdrant server URL | ✅ |
| `QDRANT_API_KEY` | Qdrant Cloud API key | ☁️ cloud only |
| `QDRANT_COLLECTION` | Collection name | default: medmitra_knowledge |
| `GROQ_API_KEY` | Groq LLM API key | Optional (LLM disabled if missing) |
| `LLM_MODEL` | LLM model name | default: llama-3.3-70b-versatile |
| `MEDCPT_QUERY_MODEL` | MedCPT query encoder | default: ncbi/MedCPT-Query-Encoder |
| `MEDCPT_ARTICLE_MODEL` | MedCPT article encoder | default: ncbi/MedCPT-Article-Encoder |
| `DENSE_TOP_K` | Dense retrieval top-k | default: 15 |
| `SPARSE_TOP_K` | Sparse retrieval top-k | default: 15 |
| `FINAL_TOP_K` | Final results after RRF | default: 5 |
| `RRF_K` | RRF constant | default: 60 |
| `KAGGLE_USERNAME` | Kaggle username | For download automation |
| `KAGGLE_KEY` | Kaggle API key | For download automation |

---

## Architecture

```
User Query
    │
    ▼
Emergency Check (deterministic keyword rules)
    │ no emergency
    ▼
Normalize + Typo Correction (RapidFuzz)
    │
    ▼
MedCPT Query Encoder → dense vector (768d)
BM25 fastembed      → sparse vector
    │
    ▼
Qdrant hybrid_search (filtered by record_type)
    │
    ▼
Reciprocal Rank Fusion (RRF k=60)
    │
    ▼
Top-K results → LLM grounding (Groq)
    │
    ▼
Response with source metadata + disclaimer
```

---

## Future Enhancements (Out of Scope for Prototype)

- **Multilingual support**: Translation models for Hindi/Tamil/Telugu queries
- **Voice assistant integration**: Whisper STT + TTS output
- **OCR prescription parsing**: Extract medicine names from prescription images
- **Verified clinical data**: Replace Kaggle datasets with WHO/ICD/NLM sources
- **Local LLM**: Qwen3.5-4B running on local GPU as offline fallback
- **250k+ medicine dataset**: Scale to larger datasets after compact baseline is validated
- **PostgreSQL integration**: Persistent session history and audit trail
- **Firebase**: Mobile push notifications and user analytics

---

## Disclaimer

> This service is an **informational prototype** only. All disease and medicine data is sourced from Kaggle datasets with `review_status = "prototype_unverified"`. This data has **not been clinically validated** by a medical professional or official health authority. The service does **not diagnose diseases**, **prescribe medicines**, or provide **clinical advice**. Always consult a qualified healthcare professional for medical decisions.
