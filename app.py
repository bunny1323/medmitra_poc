"""
app.py — MedMitra Streamlit Prototype
──────────────────────────────────────
Launch with: python -m streamlit run app.py

What this app does NOT do at startup:
  ✗ Generate embeddings
  ✗ Clean data
  ✗ Rebuild indexes

What this app DOES at startup:
  ✓ Load the existing ChromaDB collections
  ✓ Load the embedding model (from local cache)
  ✓ Check whether Groq API key is configured

Search flow per query:
  1. Emergency detection (keyword rules — instant)
  2. If emergency → show alert, stop
  3. Embed query → search ChromaDB collection
  4. If relevance < threshold → show "not found" message
  5. If relevance ≥ threshold → call Llama (if Groq key set)
  6. Display results + JSON preview
"""

import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")

# ── Project imports ───────────────────────────────────────────────────────────
from app.services.emergency_service import check_emergency
from app.services.retrieval_service  import (
    search, collection_exists,
    DISEASE_COLLECTION, MEDICINE_COLLECTION,
    get_model,
)
from app.services.llm_service        import generate_response, is_groq_configured

# ── Config ────────────────────────────────────────────────────────────────────
MIN_RELEVANCE   = float(os.getenv("MIN_RELEVANCE_SCORE", "0.55"))
DEFAULT_TOP_K   = int(os.getenv("DEFAULT_TOP_K", "5"))
APP_VERSION     = "0.1.0-poc"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedMitra — Medical Search Prototype",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached model load (once per Streamlit session) ────────────────────────────
@st.cache_resource(show_spinner="Loading medical embedding model …")
def load_embedding_model():
    """
    Load the SentenceTransformer model once.
    st.cache_resource keeps it alive across reruns.
    Streamlit does NOT rebuild the index — it only loads the model.
    """
    return get_model()


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar() -> tuple[str, int]:
    st.sidebar.title("⚕️ MedMitra")
    st.sidebar.caption(f"Version {APP_VERSION} · Prototype")
    st.sidebar.divider()

    search_mode = st.sidebar.radio(
        "Search mode",
        ["🩺 Disease / Symptom Search", "💊 Medicine Search"],
        index=0,
    )

    top_k = st.sidebar.slider(
        "Results to retrieve (top-k)",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
        help="How many vector search results to fetch from ChromaDB.",
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Model**")
    st.sidebar.code("NeuML/pubmedbert-base-embeddings", language=None)

    st.sidebar.markdown("**Llama via Groq**")
    if is_groq_configured():
        llama_model = os.getenv("LLAMA_MODEL", "llama-3.3-70b-versatile")
        st.sidebar.success(f"✅ Configured\n{llama_model}")
    else:
        st.sidebar.warning("⚠️ Not configured\nAdd GROQ_API_KEY to .env")

    st.sidebar.markdown("**Index status**")
    d_ok = collection_exists(DISEASE_COLLECTION)
    m_ok = collection_exists(MEDICINE_COLLECTION)
    st.sidebar.markdown(f"{'✅' if d_ok else '❌'} Disease index")
    st.sidebar.markdown(f"{'✅' if m_ok else '❌'} Medicine index")

    if not d_ok or not m_ok:
        st.sidebar.error(
            "One or more indexes are missing.\n\n"
            "Run these commands:\n"
            "```\npython scripts\\build_disease_index.py --reset\n"
            "python scripts\\build_medicine_index.py --reset\n```"
        )

    st.sidebar.divider()
    st.sidebar.caption(
        "⚕️ MedMitra is a **prototype** for demonstration only. "
        "It does not provide a confirmed diagnosis and does not replace "
        "a qualified healthcare professional."
    )

    return search_mode, top_k


# ── Main UI ───────────────────────────────────────────────────────────────────
def render_main(search_mode: str, top_k: int):
    st.title("⚕️ MedMitra — Medical Information Search")
    st.caption(
        "Prototype · Symptom-based disease retrieval and medicine information. "
        "Not a diagnostic tool."
    )

    # Determine collection
    if "Disease" in search_mode:
        collection   = DISEASE_COLLECTION
        placeholder  = "e.g. fever, cough, body pain and weakness"
        index_label  = "disease"
    else:
        collection   = MEDICINE_COLLECTION
        placeholder  = "e.g. medicine for fever, antibiotic for infection"
        index_label  = "medicine"

    # ── Check index exists ────────────────────────────────────────────────────
    if not collection_exists(collection):
        st.error(
            f"❌ The **{index_label}** index is not built yet.\n\n"
            f"Open PowerShell and run:\n\n"
            f"```powershell\n"
            f"python scripts\\build_{index_label}_index.py --reset\n"
            f"```"
        )
        return

    # ── Query input ───────────────────────────────────────────────────────────
    query = st.text_input(
        "Describe symptoms or search for a medicine:",
        placeholder=placeholder,
        max_chars=500,
        key="query_input",
    )

    search_btn = st.button("🔍 Search", type="primary", use_container_width=True)

    if not search_btn or not query.strip():
        st.info("Enter a query above and click Search.")
        return

    query = query.strip()
    st.divider()

    # ── Step 1: Emergency detection ───────────────────────────────────────────
    emergency = check_emergency(query)
    if emergency:
        st.error(emergency["message"])
        st.warning(emergency["footer"])

        with st.expander("Emergency detection details"):
            st.json({
                "is_emergency":    emergency["is_emergency"],
                "rule_id":         emergency["rule_id"],
                "severity":        emergency["severity"],
                "matched_phrases": emergency["matched_phrases"],
            })

        _show_json_preview({
            "query":        query,
            "is_emergency": True,
            "rule_id":      emergency["rule_id"],
            "severity":     emergency["severity"],
            "answer":       emergency["message"],
            "source":       "emergency_rules",
        })
        return  # Stop — do NOT call vector search or Llama for emergencies

    # ── Step 2: Vector retrieval ──────────────────────────────────────────────
    with st.spinner("Searching medical database …"):
        result = search(query=query, collection_name=collection, top_k=top_k)

    if "error" in result:
        st.error(f"Search error: {result['error']}")
        return

    top_relevance = result["top_relevance"]
    results       = result["results"]

    # ── Step 3: Relevance gate ────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Query:** {query}")
    with col2:
        relevance_color = (
            "🟢" if top_relevance >= 0.65
            else "🟡" if top_relevance >= MIN_RELEVANCE
            else "🔴"
        )
        st.metric(
            "Top relevance",
            f"{relevance_color} {top_relevance:.3f}",
            help=(
                "Cosine similarity between your query and the best matching record. "
                "This is NOT a diagnosis probability."
            ),
        )

    if top_relevance < MIN_RELEVANCE:
        st.warning(
            f"I could not find sufficiently relevant medical information for this query "
            f"(relevance: {top_relevance:.2f}, threshold: {MIN_RELEVANCE:.2f}).\n\n"
            "Please consult a qualified healthcare professional."
        )
        _show_disclaimer()
        _show_json_preview({
            "query":           query,
            "collection":      collection,
            "top_relevance":   top_relevance,
            "threshold":       MIN_RELEVANCE,
            "answer":          "Insufficient relevance — no result returned.",
            "results_preview": [],
        })
        return

    # ── Step 4: Show retrieved results ────────────────────────────────────────
    st.markdown("### 📋 Retrieved Records")
    st.caption(
        "⚕️ These records are retrieved by semantic similarity. "
        "They do **not** confirm any diagnosis."
    )

    for r in results:
        relevance = r["relevance_score"]
        bar_color = "#2ecc71" if relevance >= 0.65 else "#f39c12" if relevance >= MIN_RELEVANCE else "#e74c3c"

        with st.expander(
            f"Rank {r['rank']} — "
            f"{r['metadata'].get('title', 'Unknown').title()} "
            f"(relevance: {relevance:.3f})",
            expanded=(r["rank"] == 1),
        ):
            # Relevance bar
            st.markdown(
                f'<div style="background:#eee;border-radius:4px;height:8px;width:100%">'
                f'<div style="background:{bar_color};width:{min(relevance*100,100):.0f}%;'
                f'height:8px;border-radius:4px"></div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Content:** {r['content']}")

            meta = r["metadata"]
            meta_cols = st.columns(3)
            with meta_cols[0]:
                st.markdown(f"**Source type:** `{meta.get('source_type','—')}`")
            with meta_cols[1]:
                st.markdown(f"**Dataset:** `{meta.get('source_dataset','—')}`")
            with meta_cols[2]:
                st.markdown(f"**Distance:** `{r['distance']:.4f}`")

    # ── Step 5: Llama response ────────────────────────────────────────────────
    st.markdown("### 🤖 Generated Response")

    with st.spinner("Generating response …"):
        llm_result = generate_response(query=query, retrieval_result=result)

    answer = llm_result["answer"]
    source = llm_result["source"]

    if source == "groq":
        st.success(answer)
        st.caption(f"Generated by: `{llm_result['model_used']}`")
    elif llm_result.get("skipped") and "Groq API not configured" in answer:
        # Show the context-based fallback cleanly
        st.info(answer)
        st.caption("Groq API not configured — showing retrieved context summary.")
    else:
        st.info(answer)
        if llm_result.get("skip_reason"):
            st.caption(f"ℹ️ Llama skipped: {llm_result['skip_reason']}")

    _show_disclaimer()

    # ── Step 6: JSON preview (for Express → FastAPI integration) ──────────────
    _show_json_preview({
        "query":           query,
        "collection":      collection,
        "is_emergency":    False,
        "top_relevance":   top_relevance,
        "threshold":       MIN_RELEVANCE,
        "llm_source":      source,
        "llm_model":       llm_result.get("model_used"),
        "answer":          answer,
        "results_preview": [
            {
                "rank":           r["rank"],
                "title":          r["metadata"].get("title"),
                "relevance_score": r["relevance_score"],
                "source_type":    r["metadata"].get("source_type"),
            }
            for r in results[:3]
        ],
    })


def _show_disclaimer():
    st.divider()
    st.warning(
        "⚕️ **Medical Disclaimer:** MedMitra is a prototype for research and "
        "demonstration purposes only. It does not provide a confirmed diagnosis, "
        "does not prescribe treatment, and is not a substitute for professional "
        "medical advice. Always consult a qualified healthcare professional."
    )


def _show_json_preview(payload: dict):
    with st.expander("🔌 JSON — Backend Integration Preview", expanded=False):
        st.caption(
            "This is what the future FastAPI microservice will return to the Express backend."
        )
        st.json(payload)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    # Warm up the embedding model (cached — only happens once per session)
    load_embedding_model()

    search_mode, top_k = render_sidebar()
    render_main(search_mode, top_k)


if __name__ == "__main__":
    main()
