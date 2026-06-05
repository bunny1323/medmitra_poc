from typing import Dict, Any, List
from app.services.typo_handler import TypoHandler
from app.services.emergency_detector import EmergencyDetector
from app.services.medcpt_retriever import MedCPTRetriever
from app.services.bm25_retriever import BM25Retriever
from app.services.rrf_fusion import RRFFusion
from app.core.config import settings
from app.core.logging import logger

class HybridRetriever:
    def __init__(
        self,
        typo_handler: TypoHandler,
        emergency_detector: EmergencyDetector,
        medcpt_retriever: MedCPTRetriever,
        bm25_retriever: BM25Retriever,
        rrf_fusion: RRFFusion
    ):
        self.typo_handler = typo_handler
        self.emergency_detector = emergency_detector
        self.medcpt_retriever = medcpt_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_fusion = rrf_fusion

    def retrieve_hybrid(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        # 1. Normalize and typo correct
        corrected_query = self.typo_handler.correct_query(query)
        logger.info(f"Original Query: '{query}' -> Corrected Query: '{corrected_query}'")
        
        # 2. Emergency symptom detection (on normalized/corrected query)
        em_res = self.emergency_detector.check_emergency(corrected_query)
        
        emergency_detected = em_res.is_emergency
        emergency_message = None
        if emergency_detected:
            # Construct emergency message warning before results
            action_tips = [m.action for m in em_res.matches]
            action_text = " ".join(action_tips) if action_tips else "Please seek immediate medical attention."
            emergency_message = f"URGENT WARNING: {em_res.message} {action_text} If you are experiencing life-threatening symptoms, please contact emergency medical services or visit the nearest hospital immediately."

        # 3. Dense retrieval using MedCPT (using settings.DENSE_TOP_K)
        dense_top_k = getattr(settings, "DENSE_TOP_K", 10)
        dense_results = self.medcpt_retriever.retrieve(corrected_query, top_k=dense_top_k)
        
        # 4. Sparse retrieval using BM25 (using settings.BM25_TOP_K)
        bm25_top_k = getattr(settings, "BM25_TOP_K", 10)
        bm25_results = self.bm25_retriever.retrieve(corrected_query, top_k=bm25_top_k)
        
        # 5. RRF fusion (using settings.RRF_K and the requested top_k from the request)
        fused_results = self.rrf_fusion.fuse(dense_results, bm25_results, top_k=top_k)
        
        return {
            "emergency_detected": emergency_detected,
            "emergency_message": emergency_message,
            "results": fused_results
        }
