# ⚕️ MedMitra — RAG Backend API

A medical information search backend built with **FastAPI**, **Qdrant (Hybrid Search)**, **PyMuPDF**, and **Groq (Llama)**. 

This API ingests a medical textbook (`current-medical-diagnosis-and-treatment-2025-1.pdf`), chunks it, and provides an empathetic conversational interface that dynamically assesses the **severity index** of a patient's query.

---

## 🏗 Architecture

```text
Medical Book (PDF)
      ↓
PyMuPDF Extraction & LangChain Chunking
      ↓
Dense Embeddings (PubMedBERT) + Sparse Embeddings (SPLADE)
      ↓
Qdrant Vector Database (Hybrid Search / RRF)
      ↓
FastAPI Backend (/api/v1/query)
      ↓
Groq (Llama 3) generates empathetic response + Severity Index
      ↓
JSON Response to Frontend / Postman
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.11+
- Git

### 2. Install Dependencies
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 3. Add the Medical Book
Ensure `current-medical-diagnosis-and-treatment-2025-1.pdf` is present in the root directory.

### 4. Configure Environment Variables
Copy the example environment file and add your Groq API key:
```powershell
copy .env.example .env
# Edit .env and add GROQ_API_KEY=your_key_here
```

### 5. Ingest the PDF into Qdrant
Run the ingestion script. This will parse the PDF, generate hybrid embeddings, and store them in the `qdrant_db` folder.
```powershell
python scripts\ingest_medical_book.py
```

### 6. Start the FastAPI Server
```powershell
uvicorn app.main:app --reload
```
The server will run at `http://127.0.0.1:8000`.

---

## 🧪 Testing with Postman

You can test the API using Postman or cURL. 

### Endpoint Details
- **URL**: `http://127.0.0.1:8000/api/v1/query`
- **Method**: `POST`
- **Headers**: `Content-Type: application/json`

### Request Body (JSON)
```json
{
  "query": "I have a severe headache and neck stiffness",
  "top_k": 3
}
```

### Expected Response
```json
{
  "query": "I have a severe headache and neck stiffness",
  "severity_index": "URGENT",
  "confidence_score": 0.88,
  "answer": "I'm sorry you are experiencing this. A severe headache accompanied by neck stiffness could suggest a serious condition like meningitis. Please seek medical attention immediately.",
  "sources": [
    {
      "page": "125",
      "content": "Symptoms of meningitis typically include a severe headache, neck stiffness..."
    }
  ],
  "error": null
}
```

## Severity Index Levels
The LLM dynamically evaluates the query and returns one of the following:
- `NORMAL`: Routine queries, general information, minor symptoms.
- `URGENT`: Symptoms requiring prompt medical attention.
- `CRITICAL`: Life-threatening situations needing immediate emergency care.
