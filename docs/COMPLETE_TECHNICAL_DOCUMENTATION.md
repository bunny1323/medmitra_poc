# MedMitra ML/Search Service — Complete Technical Documentation

## 1. Executive Summary & Purpose
**MedMitra ML Search Service** is a stateless, production-ready backend microservice engineered to provide safe, guideline-grounded, symptom-based educational information and emergency keyword screening. Designed to support modern digital pharmacy and healthcare applications, the service serves as an educational assistant. 

It is **NOT** a diagnostic tool. To preserve clinical safety, MedMitra is prohibited from diagnosing patients, prescribing medications, advising self-treatment, or recommending dosage modifications.

---

## 2. Safety Boundaries & Protocols
Clinical safety is hardcoded at all layers of the application:
1. **Infant Safety Limit**: Symptoms in neonates under **2 months of age** are highly critical. The query classifier intercepts these requests immediately and raises a safety exception redirecting parents to immediate clinical pediatric evaluation.
2. **Emergency Escalation**: High-risk terms (e.g. chest pressure, fainting, wanting to end life) bypass semantic searches and LLMs entirely, returning instant warning escalations.
3. **No Hallucination**: Generative summaries are strictly grounded in retrieved standard treatment guidelines. If retrieval relevance is weak, the LLM falls back to degraded educational notices rather than fabricating conditions.
4. **Vocabulary Limits**: Metrics like similarity distances are exposed strictly as `retrieval_relevance` and never as diagnostic certainty or disease probability percentages.

---

## 3. Architecture & Data Flow
The service is structured as a stateless, containerized FastAPI process:
- **API Router**: Handles validation, security verification, and routes endpoints.
- **Emergency Service**: Performs deterministic checks before vector lookups.
- **Ingestion Pipeline**: Manages PyMuPDF document extractions, token-aware parsing, PubMedBERT dense vectors, FastEmbed BM25 sparse vectors, and Qdrant database writing.
- **RAG Service**: Merges hybrid retrieval outputs, validates JSON output schemas, and injects safe formatting guidelines into Groq.

---

## 4. Guidelines & Ingestion Strategy
For grounding, the system uses three official medical publications:
1. *ICMR Standard Treatment Workflows of India — Volume I* (ICMR and Department of Health Research, Government of India)
2. *ICMR Standard Treatment Workflows of India — Volume III* (ICMR and Department of Health Research, Government of India)
3. *WHO Integrated Management of Childhood Illness — Chart Booklet* (World Health Organization)

### Parsing & Chunking
- Chunks target a **260-token length** with a **45-token overlap** using PubMedBERT's WordPiece boundaries.
- Splitting respects document hierarchies (paragraphs first, then sentences, and sliding windows as a last resort).
- Source pages, sections, and authority tags are preserved for grounded citations.

---

## 5. Embeddings & Search Mechanics
- **Dense Vectors**: `NeuML/pubmedbert-base-embeddings` generates 768-dimensional vectors optimized for clinical semantics.
- **Sparse Vectors**: BM25 lexical weights map keyword frequencies to ensure exact salt name matching.
- **Ranking**: Reciprocal Rank Fusion (RRF) combines dense and sparse rankings to provide high-precision hybrid retrieval.
