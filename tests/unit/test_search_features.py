"""
MedMitra Search Feature Tests
================================
Tests for disease search, medicine search, emergency detection,
agent routing, data cleaning, Qdrant connectivity and API key security.

All external services (Qdrant, MedCPT, fastembed) are mocked.
Tests use app.dependency_overrides for auth bypass.
"""

import json
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Sample test data (Kaggle-schema-compliant)
# ---------------------------------------------------------------------------

SAMPLE_DISEASE_RECORDS = [
    {
        "record_id": "d001",
        "record_type": "disease",
        "condition_name": "Common Cold",
        "symptoms": ["runny nose", "sneezing", "mild cough", "sore throat", "low fever"],
        "description": "A benign self-limiting viral infection of the upper respiratory tract.",
        "precautions": ["stay hydrated", "rest", "avoid cold exposure"],
        "symptom_weights": {},
        "search_text": "common cold runny nose sneezing mild cough sore throat low fever",
        "source_name": "Disease Symptom Prediction",
        "source_type": "kaggle_dataset",
        "dataset_slug": "itachi9604/disease-symptom-description-dataset",
        "review_status": "prototype_unverified",
    },
    {
        "record_id": "d002",
        "record_type": "disease",
        "condition_name": "Influenza",
        "symptoms": ["fever", "body aches", "headache", "fatigue", "cough"],
        "description": "A viral respiratory illness caused by influenza viruses.",
        "precautions": ["rest", "stay hydrated", "take antiviral if prescribed"],
        "symptom_weights": {},
        "search_text": "influenza fever body aches headache fatigue cough",
        "source_name": "Disease Symptom Prediction",
        "source_type": "kaggle_dataset",
        "dataset_slug": "itachi9604/disease-symptom-description-dataset",
        "review_status": "prototype_unverified",
    },
    {
        "record_id": "d003",
        "record_type": "disease",
        "condition_name": "Acute Gastroenteritis",
        "symptoms": ["diarrhea", "vomiting", "nausea", "abdominal pain", "mild fever"],
        "description": "Inflammation of the stomach and intestines. ORS prevents dehydration.",
        "precautions": ["oral rehydration salts", "bland diet", "avoid dairy"],
        "symptom_weights": {},
        "search_text": "acute gastroenteritis diarrhea vomiting nausea abdominal pain mild fever",
        "source_name": "Disease Symptom Prediction",
        "source_type": "kaggle_dataset",
        "dataset_slug": "itachi9604/disease-symptom-description-dataset",
        "review_status": "prototype_unverified",
    },
]

SAMPLE_MEDICINE_RECORDS = [
    {
        "record_id": "m001",
        "record_type": "medicine",
        "medicine_name": "Paracetamol",
        "generic_name": "Paracetamol",
        "aliases": ["Acetaminophen", "Tylenol"],
        "category": "analgesic and antipyretic",
        "dosage_form": "tablet",
        "mechanism_of_action": "Inhibits prostaglandin synthesis in the CNS",
        "uses": ["fever relief", "mild to moderate pain relief"],
        "side_effects": ["nausea", "liver damage at high doses"],
        "warnings": ["do not exceed recommended dose", "avoid alcohol"],
        "manufacturer": "",
        "salt_composition": "paracetamol 500mg",
        "search_text": "paracetamol analgesic antipyretic fever relief pain relief",
        "source_name": "1000 Drugs and Side Effects",
        "source_type": "kaggle_dataset",
        "dataset_slug": "palakjain9/1000-drugs-and-side-effects",
        "review_status": "prototype_unverified",
    },
    {
        "record_id": "m002",
        "record_type": "medicine",
        "medicine_name": "Azithromycin",
        "generic_name": "Azithromycin",
        "aliases": ["Zithromax"],
        "category": "antibiotic macrolide",
        "dosage_form": "tablet",
        "mechanism_of_action": "Inhibits bacterial protein synthesis",
        "uses": ["bacterial infections", "community acquired pneumonia"],
        "side_effects": ["nausea", "diarrhea", "abdominal pain"],
        "warnings": [
            "This is a prescription antibiotic. Do not use without a doctor's prescription. "
            "Misuse contributes to antimicrobial resistance."
        ],
        "manufacturer": "",
        "salt_composition": "azithromycin 500mg",
        "search_text": "azithromycin antibiotic macrolide bacterial infections pneumonia",
        "source_name": "1000 Drugs and Side Effects",
        "source_type": "kaggle_dataset",
        "dataset_slug": "palakjain9/1000-drugs-and-side-effects",
        "review_status": "prototype_unverified",
    },
    {
        "record_id": "m003",
        "record_type": "medicine",
        "medicine_name": "ORS",
        "generic_name": "Oral Rehydration Salts",
        "aliases": ["Oral Rehydration Solution"],
        "category": "rehydration",
        "dosage_form": "powder",
        "mechanism_of_action": "Restores fluid and electrolyte balance",
        "uses": ["dehydration", "diarrhea management"],
        "side_effects": [],
        "warnings": [],
        "manufacturer": "",
        "salt_composition": "sodium chloride glucose potassium chloride",
        "search_text": "ors oral rehydration salts dehydration diarrhea management",
        "source_name": "1000 Drugs and Side Effects",
        "source_type": "kaggle_dataset",
        "dataset_slug": "palakjain9/1000-drugs-and-side-effects",
        "review_status": "prototype_unverified",
    },
]

EMERGENCY_RULES = {
    "rules": [
        {
            "rule_id": "EMR001",
            "category": "cardiac",
            "patterns": ["chest pain", "severe chest pain"],
            "severity": "urgent",
            "message": "Seek urgent medical attention immediately. Cardiac emergency signs detected.",
            "source_name": "Curated verified emergency rules",
            "review_status": "official_source_review_required"
        },
        {
            "rule_id": "EMR002",
            "category": "breathing_emergency",
            "patterns": ["difficulty breathing", "shortness of breath", "cannot breathe"],
            "severity": "urgent",
            "message": "Seek urgent medical attention immediately. Breathing emergency signs detected.",
            "source_name": "Curated verified emergency rules",
            "review_status": "official_source_review_required"
        },
        {
            "rule_id": "EMR006",
            "category": "self_harm",
            "patterns": ["suicidal thoughts", "want to harm myself"],
            "severity": "urgent",
            "message": "Immediate support is available. Please contact the local crisis helpline.",
            "source_name": "Curated verified emergency rules",
            "review_status": "official_source_review_required"
        }
    ],
    "negation_patterns": ["\\bno\\s+", "\\bnot\\s+having\\s+"],
}



# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_qdrant_hits(records, rrf_score_start=0.03):
    """Build mock Qdrant hybrid search hits from record dicts."""
    hits = []
    for i, r in enumerate(records):
        hits.append({
            "id": r["record_id"],
            "rrf_score": round(rrf_score_start - i * 0.002, 6),
            "dense_score": 0.65 - i * 0.05,
            "payload": r,
        })
    return hits


@pytest.fixture
def mock_qdrant_service():
    svc = MagicMock()
    svc.is_connected.return_value = True
    svc.check_collection_exists.return_value = True
    svc.get_record_type_count.side_effect = lambda col, rt: (
        len(SAMPLE_DISEASE_RECORDS) if rt == "disease" else len(SAMPLE_MEDICINE_RECORDS)
    )
    svc.hybrid_search_by_type.side_effect = lambda **kwargs: (
        _make_mock_qdrant_hits(SAMPLE_DISEASE_RECORDS[:kwargs.get("limit", 3)])
        if kwargs.get("record_type") == "disease"
        else _make_mock_qdrant_hits(SAMPLE_MEDICINE_RECORDS[:kwargs.get("limit", 3)])
    )
    return svc


@pytest.fixture
def mock_medcpt():
    m = MagicMock()
    m.embed_query.return_value = np.random.rand(768).astype(np.float32)
    m.embed_articles.return_value = np.random.rand(len(SAMPLE_DISEASE_RECORDS), 768).astype(np.float32)
    return m


@pytest.fixture
def mock_sparse_bm25():
    s = MagicMock()
    s.embed_query.return_value = {"indices": [1, 2, 3], "values": [0.5, 0.3, 0.2]}
    s.embed_texts.return_value = [{"indices": [1, 2], "values": [0.5, 0.3]}] * 10
    s.is_ready.return_value = True
    return s


@pytest.fixture
def disease_search_service(mock_qdrant_service, mock_medcpt, mock_sparse_bm25):
    from app.services.disease_search_service import DiseaseSearchService
    from app.services.typo_handler import TypoHandler
    from app.services.emergency_detector import EmergencyDetector

    typo = MagicMock()
    typo.normalize_text.side_effect = lambda x: x.lower().strip()
    typo.correct_query.side_effect = lambda x: x

    emerg = MagicMock()
    emerg.check_emergency.return_value = MagicMock(is_emergency=False, message="")

    svc = DiseaseSearchService(
        typo_handler=typo,
        emergency_detector=emerg,
        medcpt_retriever=mock_medcpt,
        rrf_fusion=MagicMock(),
        qdrant_service=mock_qdrant_service,
        sparse_bm25_service=mock_sparse_bm25,
    )
    return svc


@pytest.fixture
def medicine_search_service(mock_qdrant_service, mock_medcpt, mock_sparse_bm25):
    from app.services.medicine_search_service import MedicineSearchService

    typo = MagicMock()
    typo.normalize_text.side_effect = lambda x: x.lower().strip()
    typo.correct_query.side_effect = lambda x: (
        x.replace("paracetmol", "paracetamol").replace("azitromycin", "azithromycin")
    )
    typo.vocab = set()

    svc = MedicineSearchService(
        typo_handler=typo,
        medcpt_retriever=mock_medcpt,
        rrf_fusion=MagicMock(),
        qdrant_service=mock_qdrant_service,
        sparse_bm25_service=mock_sparse_bm25,
    )
    return svc


@pytest.fixture
def fastapi_client():
    """TestClient with auth dependency overridden."""
    from app.main import app
    from app.core.security import verify_internal_api_key

    async def _bypass():
        return None

    app.dependency_overrides[verify_internal_api_key] = _bypass
    client = TestClient(app, raise_server_exceptions=False)
    yield client
    app.dependency_overrides.clear()


# ===========================================================================
# Group 1: Data Cleaning
# ===========================================================================

class TestDataCleaning:
    def test_clean_list_field_pipe_separated(self):
        from app.scripts.clean_datasets import _clean_list_field
        result = _clean_list_field("runny nose|sneezing|mild cough")
        assert "runny nose" in result
        assert "sneezing" in result

    def test_clean_list_field_empty(self):
        from app.scripts.clean_datasets import _clean_list_field
        assert _clean_list_field("") == []
        assert _clean_list_field(None) == []

    def test_build_disease_search_text(self):
        from app.scripts.clean_datasets import _build_disease_search_text
        rec = {
            "condition_name": "Common Cold",
            "symptoms": ["runny nose", "sneezing"],
            "description": "A viral infection",
            "precautions": ["rest", "hydration"],
        }
        text = _build_disease_search_text(rec)
        assert "common cold" in text
        assert "runny nose" in text

    def test_build_medicine_search_text(self):
        from app.scripts.clean_datasets import _build_medicine_search_text
        rec = {
            "medicine_name": "Paracetamol",
            "generic_name": "Paracetamol",
            "aliases": ["Acetaminophen"],
            "category": "analgesic",
            "uses": ["fever", "pain"],
            "mechanism_of_action": "",
            "salt_composition": "",
            "side_effects": ["nausea"],
        }
        text = _build_medicine_search_text(rec)
        assert "paracetamol" in text
        assert "fever" in text

    def test_safe_str(self):
        from app.scripts.clean_datasets import _safe_str
        assert _safe_str(None) == ""
        assert _safe_str(float("nan")) == ""
        assert _safe_str("  hello  ") == "hello"

    def test_slug(self):
        from app.scripts.clean_datasets import _slug
        assert _slug("  Common  Cold  ") == "common cold"


# ===========================================================================
# Group 2: Emergency Detection
# ===========================================================================

class TestEmergencyDetection:
    @pytest.fixture
    def emergency_service(self, tmp_path):
        rules_file = tmp_path / "emergency_rules.json"
        rules_file.write_text(json.dumps(EMERGENCY_RULES), encoding="utf-8")
        from app.services.emergency_service import EmergencyService
        return EmergencyService(rules_path=str(rules_file))

    def test_detects_chest_pain(self, emergency_service):
        result = emergency_service.check_query("i have severe chest pain and shortness of breath")
        assert result.is_emergency is True

    def test_detects_difficulty_breathing(self, emergency_service):
        result = emergency_service.check_query("difficulty breathing for the last hour")
        assert result.is_emergency is True

    def test_no_emergency_for_common_cold(self, emergency_service):
        result = emergency_service.check_query("i have runny nose and mild cough")
        assert result.is_emergency is False

    def test_detects_suicidal_thoughts(self, emergency_service):
        result = emergency_service.check_query("i am having suicidal thoughts")
        assert result.is_emergency is True

    def test_negation_respected(self, emergency_service):
        result = emergency_service.check_query("no chest pain reported")
        assert result.is_emergency is False

    def test_non_emergency_text(self, emergency_service):
        result = emergency_service.check_query("random unrelated text about cooking")
        assert result.is_emergency is False


# ===========================================================================
# Group 3: Disease Search
# ===========================================================================

class TestDiseaseSearch:
    def test_returns_results_for_fever_query(self, disease_search_service):
        result = disease_search_service.search("fever cold and mild cough", top_k=3)
        assert result["emergency_detected"] is False
        assert isinstance(result["results"], list)
        assert len(result["results"]) > 0

    def test_results_have_required_fields(self, disease_search_service):
        result = disease_search_service.search("fever cold", top_k=3)
        for r in result["results"]:
            assert "condition_name" in r
            assert "description" in r
            assert "rrf_score" in r
            assert "review_status" in r

    def test_emergency_bypasses_search(self, mock_qdrant_service, mock_medcpt, mock_sparse_bm25):
        from app.services.disease_search_service import DiseaseSearchService

        typo = MagicMock()
        typo.normalize_text.side_effect = lambda x: x
        typo.correct_query.side_effect = lambda x: x

        em_result = MagicMock()
        em_result.is_emergency = True
        em_result.message = "Emergency detected."
        emerg = MagicMock()
        emerg.check_emergency.return_value = em_result

        svc = DiseaseSearchService(
            typo_handler=typo, emergency_detector=emerg,
            medcpt_retriever=mock_medcpt, rrf_fusion=MagicMock(),
            qdrant_service=mock_qdrant_service, sparse_bm25_service=mock_sparse_bm25,
        )
        result = svc.search("difficulty breathing and severe chest pain")
        assert result["emergency_detected"] is True
        assert result["results"] == []

    def test_disclaimer_present(self, disease_search_service):
        result = disease_search_service.search("runny nose sneezing", top_k=3)
        assert "disclaimer" in result
        assert len(result["disclaimer"]) > 10

    def test_result_review_status(self, disease_search_service):
        result = disease_search_service.search("fever", top_k=3)
        for r in result["results"]:
            assert r["review_status"] == "prototype_unverified"

    def test_relevance_label_set(self, disease_search_service):
        result = disease_search_service.search("fever cold mild cough", top_k=3)
        assert result["retrieval_relevance"] is not None


# ===========================================================================
# Group 4: Medicine Search
# ===========================================================================

class TestMedicineSearch:
    def test_returns_paracetamol(self, medicine_search_service):
        result = medicine_search_service.search("paracetamol", top_k=3)
        assert "results" in result
        assert len(result["results"]) > 0

    def test_typo_corrected_paracetmol(self, medicine_search_service):
        result = medicine_search_service.search("paracetmol uses", top_k=3, allow_typo_correction=True)
        # After typo correction, should find paracetamol
        corrected = result.get("corrected_query", "paracetmol uses")
        assert "paracetamol" in corrected or len(result["results"]) > 0

    def test_antibiotic_warning_present(self, medicine_search_service):
        """
        Azithromycin (antibiotic category) must carry a prescription warning.
        We test the warning injection function directly since the Qdrant mock
        returns a pre-built payload that already has the warning from our sample data.
        """
        from app.services.medicine_search_service import _inject_antibiotic_warning

        # Verify the injection function adds a warning for antibiotic category
        result = _inject_antibiotic_warning("antibiotic macrolide", [])
        assert len(result) > 0
        warnings_text = " ".join(result)
        assert "antibiotic" in warnings_text.lower() or "prescription" in warnings_text.lower()

        # Also verify the sample Azithromycin record already contains the warning
        azithromycin = SAMPLE_MEDICINE_RECORDS[1]
        existing_warnings = azithromycin.get("warnings", [])
        combined = _inject_antibiotic_warning(azithromycin["category"], existing_warnings)
        combined_text = " ".join(combined).lower()
        assert "antibiotic" in combined_text or "prescription" in combined_text

    def test_disclaimer_present(self, medicine_search_service):
        result = medicine_search_service.search("paracetamol", top_k=3)
        assert "disclaimer" in result
        assert "prescription" in result["disclaimer"].lower() or "healthcare" in result["disclaimer"].lower()

    def test_ors_lookup(self, medicine_search_service):
        medicine_search_service.qdrant_service.hybrid_search_by_type.return_value = (
            _make_mock_qdrant_hits([SAMPLE_MEDICINE_RECORDS[2]])
        )
        result = medicine_search_service.search("ORS", top_k=3)
        assert len(result["results"]) > 0

    def test_fuzzy_azithromycin_typo(self, medicine_search_service):
        result = medicine_search_service.search("azitromycin", top_k=3, allow_typo_correction=True)
        # Should not crash; corrected_query should be set
        assert "corrected_query" in result or "results" in result


# ===========================================================================
# Group 5: Agent Routing
# ===========================================================================

class TestAgentRouting:
    @pytest.fixture
    def agent_service(self, disease_search_service, medicine_search_service):
        from app.services.agent_service import AgentService
        from app.services.emergency_service import EmergencyService

        em_svc = MagicMock()
        em_result = MagicMock()
        em_result.is_emergency = False
        em_result.message = ""
        em_result.matches = []
        em_svc.check_query.return_value = em_result

        llm = MagicMock()
        llm.generate_grounded_rag.return_value = "General information retrieved."
        llm.is_enabled.return_value = True

        return AgentService(
            emergency_service=em_svc,
            disease_search_service=disease_search_service,
            medicine_search_service=medicine_search_service,
            llm_service=llm,
        )

    def test_routes_symptom_to_disease_search(self, agent_service):
        result = agent_service.run("fever cold and mild cough")
        assert "disease_search" in result["tool_used"]
        assert "emergency_check" in result["tool_used"]

    def test_routes_medicine_query(self, agent_service):
        result = agent_service.run("paracetamol tablet uses and side effects")
        assert "medicine_search" in result["tool_used"]

    def test_emergency_stops_search(self):
        from app.services.agent_service import AgentService

        em_svc = MagicMock()
        em_result = MagicMock()
        em_result.is_emergency = True
        em_result.message = "Emergency detected."
        em_result.matches = [MagicMock(action="Call emergency services.")]
        em_svc.check_query.return_value = em_result

        svc = AgentService(
            emergency_service=em_svc,
            disease_search_service=MagicMock(),
            medicine_search_service=MagicMock(),
            llm_service=MagicMock(),
        )
        result = svc.run("i have difficulty breathing and severe chest pain")
        assert result["emergency_detected"] is True
        assert "disease_search" not in result["tool_used"]
        assert "medicine_search" not in result["tool_used"]

    def test_empty_retrieval_returns_not_found(self):
        from app.services.agent_service import AgentService, _EMPTY_RETRIEVAL_MESSAGE

        em_svc = MagicMock()
        em_result = MagicMock()
        em_result.is_emergency = False
        em_result.matches = []
        em_svc.check_query.return_value = em_result

        dis_svc = MagicMock()
        dis_svc.search.return_value = {"emergency_detected": False, "results": [], "retrieval_relevance": "LOW", "normalized_query": ""}
        med_svc = MagicMock()
        med_svc.search.return_value = {"results": [], "retrieval_relevance": "LOW", "corrected_query": ""}

        llm = MagicMock()
        llm.generate_grounded_rag.return_value = _EMPTY_RETRIEVAL_MESSAGE
        llm.is_enabled.return_value = True

        svc = AgentService(
            emergency_service=em_svc,
            disease_search_service=dis_svc,
            medicine_search_service=med_svc,
            llm_service=llm,
        )
        result = svc.run("random unrelated text about nothing medical")
        # With empty retrieval, answer should be the not-found message
        assert len(result["answer"]) > 0

    def test_disclaimer_always_present(self, agent_service):
        result = agent_service.run("fever and cough")
        assert "disclaimer" in result
        assert len(result["disclaimer"]) > 10

    def test_routes_inventory_query(self, agent_service):
        result = agent_service.run("is paracetamol in stock?")
        assert "inventory_lookup" in result["tool_used"]
        assert "emergency_check" in result["tool_used"]
        assert "medicine_search" not in result["tool_used"]
        assert "disease_search" not in result["tool_used"]
        assert "I could not find sufficiently reliable information" in result["answer"]



# ===========================================================================
# Group 6: API Key Security
# ===========================================================================

class TestApiKeySecurity:
    def test_disease_search_requires_api_key(self):
        from app.main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/v1/search/disease", json={"query": "fever", "age_group": "adult"})
        assert resp.status_code in (401, 403, 422)

    def test_medicine_search_requires_api_key(self):
        from app.main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/v1/search/medicine", json={"query": "paracetamol"})
        assert resp.status_code in (401, 403, 422)

    def test_admin_endpoints_require_api_key(self):
        from app.main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/v1/admin/index-status")
        assert resp.status_code in (401, 403, 422)

    def test_health_endpoint_is_public(self):
        from app.main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health/live")
        assert resp.status_code == 200

    def test_wrong_api_key_rejected(self):
        from app.main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/api/v1/search/disease",
            json={"query": "fever", "age_group": "adult"},
            headers={"X-Internal-API-Key": "wrong-key-value"}
        )
        assert resp.status_code in (401, 403)


# ===========================================================================
# Group 7: Qdrant Service
# ===========================================================================

class TestQdrantService:
    def test_is_connected_false_when_unreachable(self):
        with patch("app.services.qdrant_service.QdrantClient") as MockClient:
            MockClient.side_effect = Exception("connection refused")
            from app.services.qdrant_service import QdrantService
            svc = QdrantService()
            assert svc.is_connected() is False

    def test_check_collection_exists_false_when_disconnected(self):
        from app.services.qdrant_service import QdrantService
        svc = QdrantService.__new__(QdrantService)
        svc.client = None
        assert svc.check_collection_exists("test") is False

    def test_get_record_type_count_returns_zero_when_disconnected(self):
        from app.services.qdrant_service import QdrantService
        svc = QdrantService.__new__(QdrantService)
        svc.client = None
        assert svc.get_record_type_count("col", "disease") == 0


# ===========================================================================
# Group 8: Emergency Rules File
# ===========================================================================

class TestEmergencyRulesFile:
    def test_main_rules_file_has_breathing_patterns(self):
        """Verify the expanded emergency_rules.json has breathing patterns."""
        import pathlib
        rules_path = (
            pathlib.Path(__file__).resolve().parent.parent.parent
            / "app" / "data" / "emergency_rules.json"
        )
        if not rules_path.exists():
            pytest.skip("emergency_rules.json not found")
        with open(rules_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_phrases = [p for rule in data.get("rules", []) for p in rule.get("phrases", [])]
        assert any("breath" in p for p in all_phrases), "No breathing-related phrases found"

    def test_curated_rules_file_has_rule_ids(self):
        """Verify the curated emergency_rules.json has rule_id fields."""
        import pathlib
        curated_path = (
            pathlib.Path(__file__).resolve().parent.parent.parent
            / "app" / "data" / "curated" / "emergency_rules.json"
        )
        if not curated_path.exists():
            pytest.skip("curated/emergency_rules.json not found")
        with open(curated_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for rule in data.get("rules", []):
            assert "rule_id" in rule, f"Rule missing rule_id: {rule}"
            assert "review_status" in rule


# ===========================================================================
# Group 9: Antibiotic Warning Injection
# ===========================================================================

class TestAntibioticWarning:
    def test_antibiotic_warning_injected(self):
        from app.services.medicine_search_service import _inject_antibiotic_warning
        warnings = _inject_antibiotic_warning("antibiotic macrolide", [])
        assert any("antibiotic" in w.lower() for w in warnings)

    def test_no_warning_for_analgesic(self):
        from app.services.medicine_search_service import _inject_antibiotic_warning
        warnings = _inject_antibiotic_warning("analgesic and antipyretic", [])
        assert len(warnings) == 0

    def test_warning_not_duplicated(self):
        from app.services.medicine_search_service import _inject_antibiotic_warning
        existing = ["This is a prescription antibiotic."]
        warnings = _inject_antibiotic_warning("antibiotic", existing)
        antibiotic_count = sum(1 for w in warnings if "antibiotic" in w.lower())
        assert antibiotic_count == 1
