# Troubleshooting Guide

This guide describes how to resolve common issues encountered during setup, ingestion, or execution of the **MedMitra ML Search microservice**.

---

## 1. Vector Database Issues

### Error: Qdrant Port Conflict
```
Bind for 0.0.0.0:6333 failed: port is already allocated
```
**Cause**: Another service or container is already utilizing port `6333`.

**Solutions**:
1. Check running containers:
   ```bash
   docker ps
   ```
2. If another instance of Qdrant is running, you can stop it:
   ```bash
   docker stop <container-name>
   ```
3. Or change the host port binding inside your `.env`:
   ```env
   QDRANT_HOST_PORT=6334
   ```
   Rerun `docker compose up -d qdrant`. The local port will map to `6334`.

---

## 2. Ingestion & File Validation Failures

### Error: "Missing required medical PDFs"
```
Missing required medical PDFs.
Required folder: data/books/
```
**Cause**: The required clinical guidelines are missing from `data/books/` or filenames do not match.

**Solutions**:
1. Check `docs/BOOK_SOURCES.md` for verified download URLs.
2. Ensure files are saved exactly as:
   - `icmr_stw_volume_1.pdf`
   - `icmr_stw_volume_3.pdf`
   - `who_imci_chart_booklet.pdf`
3. Rerun:
   ```bash
   python -m scripts.validate_pdfs
   ```

---

## 3. LLM & API Integration Errors

### Status: "degraded" or "not_ready" Health Check
**Cause**: `GROQ_API_KEY` is missing, or Qdrant connection is down.

**Solutions**:
1. Verify `GROQ_API_KEY` is set inside `.env`.
2. Confirm the key has active credit and is not expired by hitting the test endpoint.
3. If Groq is missing, the service operates in **degraded mode** using local rule-matching.

### Error: Malformed JSON Fallback
**Cause**: Groq Llama 70B outputted invalid JSON format.

**Solutions**:
1. The `LlmService` automatically catches parse errors and falls back to a safe, static educational format listing citations.
2. Set `LLM_TEMPERATURE=0.15` to ensure output remains deterministic and matches schema configurations.
