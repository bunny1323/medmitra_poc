from app.services.emergency_service import EmergencyService
from app.services.embedding_service import EmbeddingService
from app.services.sparse_embedding_service import SparseEmbeddingService
from app.services.qdrant_service import QdrantService
from app.services.query_classifier import QueryClassifier
from app.services.llm_service import LlmService
from app.services.retrieval_service import RetrievalService
from app.services.rag_service import RagService
from app.services.ocr_service import OcrService
from app.services.typo_service import TypoService



class ServiceContainer:
    def __init__(self):
        # Core infrastructure
        self._emergency_service = None
        self._embedding_service = None
        self._sparse_embedding_service = None
        self._qdrant_service = None
        self._query_classifier = None
        self._llm_service = None
        self._retrieval_service = None
        self._rag_service = None
        self._ocr_service = None
        self._typo_service = None

        # New hybrid retrieval components
        self._medcpt_retriever = None
        self._bm25_retriever = None
        self._rrf_fusion = None
        self._emergency_detector = None
        self._typo_handler = None
        self._hybrid_retriever = None
        self._sparse_bm25_service = None

        # Search services
        self._disease_search_service = None
        self._medicine_search_service = None

        # Agent
        self._agent_service = None

    # ------------------------------------------------------------------
    # Core services
    # ------------------------------------------------------------------

    @property
    def emergency_service(self):
        if not self._emergency_service:
            self._emergency_service = EmergencyService()
        return self._emergency_service


    @property
    def embedding_service(self):
        if not self._embedding_service:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    @property
    def sparse_embedding_service(self):
        if not self._sparse_embedding_service:
            self._sparse_embedding_service = SparseEmbeddingService()
        return self._sparse_embedding_service

    @property
    def qdrant_service(self):
        if not self._qdrant_service:
            self._qdrant_service = QdrantService()
        return self._qdrant_service


    @property
    def query_classifier(self):
        if not self._query_classifier:
            self._query_classifier = QueryClassifier()
        return self._query_classifier

    @property
    def llm_service(self):
        if not self._llm_service:
            self._llm_service = LlmService()
        return self._llm_service

    @property
    def retrieval_service(self):
        if not self._retrieval_service:
            self._retrieval_service = RetrievalService(
                self.embedding_service, self.sparse_embedding_service, self.qdrant_service
            )
        return self._retrieval_service

    @property
    def rag_service(self):
        if not self._rag_service:
            self._rag_service = RagService(
                self.emergency_service, self.query_classifier,
                self.retrieval_service, self.llm_service
            )
        return self._rag_service

    @property
    def ocr_service(self):
        if not self._ocr_service:
            self._ocr_service = OcrService()
        return self._ocr_service

    @property
    def typo_service(self):
        if not self._typo_service:
            self._typo_service = TypoService()
        return self._typo_service

    # ------------------------------------------------------------------
    # Hybrid retrieval components
    # ------------------------------------------------------------------

    @property
    def medcpt_retriever(self):
        if not self._medcpt_retriever:
            from app.services.medcpt_retriever import MedCPTRetriever
            self._medcpt_retriever = MedCPTRetriever()
        return self._medcpt_retriever

    @property
    def bm25_retriever(self):
        if not self._bm25_retriever:
            from app.services.bm25_retriever import BM25Retriever
            self._bm25_retriever = BM25Retriever()
        return self._bm25_retriever

    @property
    def sparse_bm25_service(self):
        if not self._sparse_bm25_service:
            from app.services.sparse_bm25_service import SparseBM25Service
            self._sparse_bm25_service = SparseBM25Service()
        return self._sparse_bm25_service

    @property
    def rrf_fusion(self):
        if not self._rrf_fusion:
            from app.services.rrf_fusion import RRFFusion
            from app.core.config import settings
            self._rrf_fusion = RRFFusion(k=settings.RRF_K)
        return self._rrf_fusion

    @property
    def emergency_detector(self):
        if not self._emergency_detector:
            from app.services.emergency_detector import EmergencyDetector
            self._emergency_detector = EmergencyDetector()
        return self._emergency_detector

    @property
    def typo_handler(self):
        if not self._typo_handler:
            from app.services.typo_handler import TypoHandler
            self._typo_handler = TypoHandler()
        return self._typo_handler

    @property
    def hybrid_retriever(self):
        if not self._hybrid_retriever:
            from app.services.hybrid_retriever import HybridRetriever
            self._hybrid_retriever = HybridRetriever(
                self.typo_handler,
                self.emergency_detector,
                self.medcpt_retriever,
                self.bm25_retriever,
                self.rrf_fusion
            )
        return self._hybrid_retriever

    # ------------------------------------------------------------------
    # Search services (Qdrant-backed)
    # ------------------------------------------------------------------

    @property
    def disease_search_service(self):
        if not self._disease_search_service:
            from app.services.disease_search_service import DiseaseSearchService
            self._disease_search_service = DiseaseSearchService(
                typo_handler=self.typo_handler,
                emergency_detector=self.emergency_detector,
                medcpt_retriever=self.medcpt_retriever,
                rrf_fusion=self.rrf_fusion,
                qdrant_service=self.qdrant_service,
                sparse_bm25_service=self.sparse_bm25_service,
            )
        return self._disease_search_service

    @property
    def medicine_search_service(self):
        if not self._medicine_search_service:
            from app.services.medicine_search_service import MedicineSearchService
            self._medicine_search_service = MedicineSearchService(
                typo_handler=self.typo_handler,
                medcpt_retriever=self.medcpt_retriever,
                rrf_fusion=self.rrf_fusion,
                qdrant_service=self.qdrant_service,
                sparse_bm25_service=self.sparse_bm25_service,
            )
        return self._medicine_search_service

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------

    @property
    def agent_service(self):
        if not self._agent_service:
            from app.services.agent_service import AgentService
            self._agent_service = AgentService(
                emergency_service=self.emergency_service,
                disease_search_service=self.disease_search_service,
                medicine_search_service=self.medicine_search_service,
                llm_service=self.llm_service,
            )
        return self._agent_service


container = ServiceContainer()
