from typing import List, Tuple, Optional
from app.core.config import settings
from app.models.enums import RelevanceLabel
from app.models.schemas import DiseaseSearchResultDetail, SourceDetail

class RetrievalService:
    def __init__(self, embedding_service, sparse_embedding_service, qdrant_service):
        self.embedding_service = embedding_service
        self.sparse_embedding_service = sparse_embedding_service
        self.qdrant_service = qdrant_service

    def retrieve(
        self,
        query: str,
        age_group: str,
        topic_group: Optional[str] = None,
        top_k: int = 4,
        include_full_text: bool = False
    ) -> Tuple[List[DiseaseSearchResultDetail], RelevanceLabel]:
        if not self.qdrant_service.is_connected():
            return [], RelevanceLabel.LOW
        try:
            dense_vec = self.embedding_service.get_dense_embeddings(query)[0]
            sparse_vec = self.sparse_embedding_service.get_sparse_embeddings(query)[0]
            
            res = self.qdrant_service.search_hybrid(
                settings.QDRANT_COLLECTION_ALIAS,
                dense_vec,
                sparse_vec,
                limit=top_k,
                age_group=age_group,
                topic_group=topic_group
            )
            
            # Apply retrieval priority boosting based on query keywords
            query_clean = query.lower()
            
            # Keywords representing core conditions (Volume I)
            v1_keywords = {
                "cough", "breath", "breathing", "respiratory", "sputum", "pneumonia", "tuberculosis", "cold", "flu",
                "ear", "nose", "throat", "sinus", "tonsil", "otitis", "pharyngitis", "congestion",
                "chest", "heart", "cardiac", "cardiology", "hypertension", "palpitation",
                "brain", "stroke", "seizure", "convulsion", "headache", "neurology", "weakness", "slurred speech",
                "kidney", "renal", "urine", "urinary", "nephrology", "dialysis"
            }
            
            # Keywords representing extended conditions (Volume III)
            v3_keywords = {
                "diabetes", "thyroid", "endocrine", "endocrinology",
                "stomach", "jaundice", "liver", "acid", "gastric", "ulcer", "gastroenterology",
                "skin", "dermatology", "rash", "itching", "eczema", "psoriasis"
            }
            
            is_v1_query = any(w in query_clean for w in v1_keywords)
            is_v3_query = any(w in query_clean for w in v3_keywords)
            
            # Priority boosts:
            # v1 default is 110, v3 default is 100, WHO child is 120
            v1_priority = 110.0
            v3_priority = 100.0
            who_priority = 120.0
            
            if is_v3_query and not is_v1_query:
                # Rank Volume III higher if query relates only to Volume III topics
                v1_priority = 100.0
                v3_priority = 120.0
            elif is_v1_query:
                # Rank Volume I higher
                v1_priority = 120.0
                v3_priority = 100.0
                
            boosted_res = []
            for pt in res:
                payload = pt["payload"]
                score = pt["score"]
                source_id = payload.get("source_id", "")
                
                boost_factor = 1.0
                if source_id == "icmr_volume_1":
                    boost_factor = v1_priority / 100.0
                elif source_id == "icmr_volume_3":
                    boost_factor = v3_priority / 100.0
                elif source_id == "who_imci":
                    boost_factor = who_priority / 100.0
                    
                pt["score"] = score * boost_factor
                boosted_res.append(pt)
                
            # Re-sort boosted results by score descending
            res = sorted(boosted_res, key=lambda x: x["score"], reverse=True)
            
            out = []
            max_score = 0.0
            for pt in res:
                payload = pt["payload"]
                score = pt["score"]
                max_score = max(max_score, score)
                
                snippet = payload["text"]
                if len(snippet) > 900:
                    snippet = snippet[:897] + "..."
                
                source = SourceDetail(
                    source_title=payload["source_title"],
                    source_file=payload["source_file"],
                    page_number=payload["page_number"],
                    section=payload.get("section", "General"),
                    age_group=payload["age_group"]
                )
                out.append(DiseaseSearchResultDetail(snippet=snippet, score=round(score, 4), source=source))
                
            relevance = RelevanceLabel.LOW
            if out:
                relevance = RelevanceLabel.HIGH if max_score >= 0.50 else RelevanceLabel.MEDIUM
            return out, relevance
        except Exception:
            return [], RelevanceLabel.LOW
