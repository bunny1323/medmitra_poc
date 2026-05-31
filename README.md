# ⚕️ MedMitra — ML / Search Prototype

A beginner-friendly, technically correct medical information search prototype
built with Streamlit, ChromaDB, sentence-transformers, and optional Llama via Groq.

> **Disclaimer:** This is a research prototype. It does not provide a confirmed
> diagnosis, does not prescribe treatment, and is not a substitute for
> professional medical advice.

---

## Architecture in Simple Words

```
Raw CSV files
  ↓
Inspect schema (scripts/inspect_datasets.py)
  ↓
Clean and validate (clean_disease_data.py, clean_medicine_data.py)
  ↓
Create prototype subsets (create_prototype_data.py)
  ↓
Build medical embeddings ONCE — offline
  (build_disease_index.py, build_medicine_index.py)
  ↓
Store vectors persistently in chroma_db/
  ↓
Launch Streamlit (app.py) — NEVER rebuilds index
  ↓
User types query → Emergency check → ChromaDB search
  ↓
If relevant → Call Llama (Groq) for safe explanation
  ↓
Show results + JSON preview for Express → FastAPI integration
```

---

## Dataset Facts (confirmed by inspection)

### Disease Dataset
| Property | Value |
|---|---|
| File | `Final_Augmented_dataset_Diseases_and_Symptoms.csv` |
| Rows | 246,945 (189,647 after dedup) |
| Format | **FORMAT A — Binary columns** (377 symptom columns, values 0 or 1) |
| Disease column | `diseases` |
| Unique diseases | 773 |

### Medicine Dataset
| Property | Value |
|---|---|
| File | `medicine_dataset.csv` |
| Rows | 50,000 |
| Unique names | 64 (SYNTHETIC dataset) |
| Columns | Name, Category, Dosage Form, Strength, Manufacturer, Indication, Classification |
| Indications | 8 (Fever, Pain, Infection, Wound, Virus, Fungus, Depression, Diabetes) |

⚠️ **The medicine dataset is synthetic** — names like `Acetocillin` do not exist.
Replace with real pharmacopoeia data (OpenFDA, WHO EML) before any clinical use.

---

## Setup — Windows PowerShell (step by step)

### Prerequisites
- Python 3.11 (from python.org)
- VS Code
- Git (optional)

### 1. Create and open project

```powershell
mkdir medmitra-streamlit-poc
cd medmitra-streamlit-poc
code .
```

### 2. Create virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks the script:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

Fallback (if activation fails):
```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> First install downloads ~1.5 GB (PyTorch CPU + sentence-transformers).
> Subsequent installs use cache.

### 4. Add CSV files

Copy these files into the `data/` folder:
- `Final_Augmented_dataset_Diseases_and_Symptoms.csv`
- `medicine_dataset.csv`

### 5. Run the pipeline

```powershell
# Step 1: Inspect datasets (always run this first)
python scripts\inspect_datasets.py

# Step 2: Clean datasets
python scripts\clean_disease_data.py
python scripts\clean_medicine_data.py

# Step 3: Create prototype subsets
python scripts\create_prototype_data.py

# Step 4: Build vector indexes (downloads model ~440 MB on first run)
python scripts\build_disease_index.py --reset
python scripts\test_disease_search.py

python scripts\build_medicine_index.py --reset
python scripts\test_medicine_search.py

# Step 5: Configure Groq (optional)
copy .env.example .env
# Edit .env — add your GROQ_API_KEY from https://console.groq.com

# Step 6: Launch Streamlit
python -m streamlit run app.py
```

---

## Collections (separate by design)

| Collection | Content | Build script |
|---|---|---|
| `medmitra_diseases` | 500 disease-symptom records | `build_disease_index.py` |
| `medmitra_medicines` | 500 medicine-indication records | `build_medicine_index.py` |

These are kept **strictly separate** to prevent disease and medicine results
from appearing in each other's searches.

---

## Emergency Detection

Emergency keywords are checked **before** any vector search.
If matched, the emergency message is shown immediately and Llama is NOT called.

Edit `config/emergency_rules.json` to add or remove phrases.

Emergency categories: cardiac, breathing, unconscious, stroke, bleeding,
seizure, overdose, mental health crisis, anaphylaxis.

---

## Llama / Groq Configuration

1. Get a free API key at https://console.groq.com
2. Copy `.env.example` to `.env`
3. Add your key: `GROQ_API_KEY=gsk_...`
4. Default model: `llama-3.3-70b-versatile`

If the key is missing or the API fails, the app shows a safe fallback —
it never crashes.

---

## Safety Rules

1. Relevance scores are **cosine similarity** — not diagnosis probabilities
2. Llama is only called if relevance ≥ 0.55 (configurable)
3. Llama is **never** called for emergency queries
4. The system prompt forbids inventing information outside the context
5. No dosage recommendations are ever made
6. A disclaimer is shown after every response

---

## GitHub Sharing

### What to commit
```
app.py
requirements.txt
.env.example       ← template only, no real key
.gitignore
README.md
app/
config/
scripts/
tests/
data/README.md     ← optional notes only
```

### What NOT to commit (already in .gitignore)
```
.env               ← contains your secret API key
venv/              ← recreated by each developer
chroma_db/         ← rebuilt from CSVs
data/*.csv         ← may contain health data
```

### Why .env must never be pushed
Your `GROQ_API_KEY` is a secret. Anyone with it can make API calls billed
to your account. The `.gitignore` already excludes `.env`. Run
`git status` before every commit to confirm it is not staged.

### How a teammate clones and runs

```powershell
git clone https://github.com/YOUR_USERNAME/medmitra-streamlit-poc
cd medmitra-streamlit-poc
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Copy CSV files into data/
python scripts\clean_disease_data.py
python scripts\clean_medicine_data.py
python scripts\create_prototype_data.py
python scripts\build_disease_index.py --reset
python scripts\build_medicine_index.py --reset
copy .env.example .env
# Add GROQ_API_KEY to .env
python -m streamlit run app.py
```

---

## Future Migration Path

```
Current:   Streamlit prototype (this repo)
              ↓
Phase 2:   FastAPI ML microservice
           - POST /search/disease  { query, top_k }
           - POST /search/medicine { query, top_k }
           - POST /emergency-check { query }
           - JSON responses match current "Backend Preview" panel
              ↓
Phase 3:   Express backend calls FastAPI via HTTP
           - Express handles auth, routing, user sessions
           - FastAPI handles embeddings, ChromaDB, Llama
           - Production ChromaDB → cloud vector DB (Pinecone, Weaviate)
           - Production LLM → fine-tuned medical model
```

The JSON shown in the "Backend Integration Preview" panel in Streamlit
is already shaped to match the FastAPI response contract.

---

## Debugging Common Errors

| Error | Fix |
|---|---|
| `Collection not found` | Run `build_disease_index.py --reset` |
| `Module not found` | Activate venv: `.\venv\Scripts\Activate.ps1` |
| `sqlite3` error on Windows | Upgrade: `pip install chromadb==0.4.24` |
| `GROQ_API_KEY not set` | App shows fallback — not an error |
| First run very slow | Model downloading (~440 MB) — wait once |
| `Execution policy` error | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
