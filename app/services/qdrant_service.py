from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from app.core.config import settings
from app.core.exceptions import DatabaseConnectionException


class QdrantService:
    """
    Qdrant vector store service.

    Supports:
    - Legacy collection management (recreate_collection, upload_points)
    - New Kaggle-dataset-based collection with named dense + sparse vectors
    - Hybrid search filtered by record_type payload field
    """

    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self._connect()

    def _connect(self):
        try:
            kwargs = {"url": settings.QDRANT_URL, "timeout": 15.0}
            if settings.QDRANT_API_KEY:
                kwargs["api_key"] = settings.QDRANT_API_KEY
            self.client = QdrantClient(**kwargs)
            self.client.get_collections()
        except Exception:
            self.client = None

    def is_connected(self) -> bool:
        if not self.client:
            return False
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def check_collection_exists(self, collection_name: str) -> bool:
        if not self.client:
            return False
        try:
            cols = self.client.get_collections()
            return any(c.name == collection_name for c in cols.collections)
        except Exception:
            return False

    def get_collection_points_count(self, collection_name: str) -> int:
        if not self.client:
            return 0
        try:
            return self.client.get_collection(collection_name).points_count or 0
        except Exception:
            return 0

    def get_record_type_count(self, collection_name: str, record_type: str) -> int:
        """Count points in collection with a specific record_type payload value."""
        if not self.client:
            return 0
        try:
            result = self.client.count(
                collection_name=collection_name,
                count_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="record_type",
                            match=qmodels.MatchValue(value=record_type)
                        )
                    ]
                ),
                exact=True
            )
            return result.count
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_or_recreate_collection(self, collection_name: str, vector_size: int = 768):
        """
        Create (or recreate) a collection with:
        - Named dense vector: 'dense' (cosine, size=vector_size)
        - Named sparse vector: 'sparse'
        """
        if not self.client:
            raise DatabaseConnectionException("Qdrant client not connected.")
        try:
            # Delete existing collection if present
            if self.check_collection_exists(collection_name):
                self.client.delete_collection(collection_name)

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": qmodels.VectorParams(
                        size=vector_size,
                        distance=qmodels.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": qmodels.SparseVectorParams(
                        index=qmodels.SparseIndexParams(on_disk=False)
                    )
                },
            )

            # Create payload index on record_type for fast filtered search
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="record_type",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
        except Exception as e:
            raise DatabaseConnectionException(f"Collection creation failed: {e}")

    # Legacy recreate (kept for backward compat with book ingestion)
    def recreate_collection(self, collection_name: str):
        if not self.client:
            raise DatabaseConnectionException()
        try:
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config={"dense": qmodels.VectorParams(size=768, distance=qmodels.Distance.COSINE)},
                sparse_vectors_config={"sparse": qmodels.SparseVectorParams()}
            )
        except Exception as e:
            raise DatabaseConnectionException(str(e))

    def create_alias(self, collection_name: str, alias_name: str):
        if not self.client:
            return
        try:
            self.client.update_collection_aliases(
                change_aliases_operations=[
                    qmodels.CreateAliasOperation(
                        create_alias=qmodels.CreateAlias(
                            collection_name=collection_name,
                            alias_name=alias_name
                        )
                    )
                ]
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Upsert records (Kaggle-dataset pipeline)
    # ------------------------------------------------------------------

    def upsert_records(
        self,
        collection_name: str,
        records: List[Dict],
        dense_vectors: List[List[float]],
        sparse_vectors: List[Dict],
        batch_size: int = 64
    ):
        """
        Upsert records with dense and sparse named vectors.

        sparse_vectors: list of {"indices": [...], "values": [...]}
        """
        if not self.client:
            raise DatabaseConnectionException("Qdrant client not connected.")

        points = []
        for i, rec in enumerate(records):
            sparse_vec = sparse_vectors[i]
            point = qmodels.PointStruct(
                id=rec["record_id"],
                vector={
                    "dense": dense_vectors[i],
                    "sparse": qmodels.SparseVector(
                        indices=sparse_vec.get("indices", []),
                        values=sparse_vec.get("values", [])
                    ),
                },
                payload={
                    "record_id": rec.get("record_id", ""),
                    "record_type": rec.get("record_type", ""),
                    "title": rec.get("condition_name") or rec.get("medicine_name") or "",
                    "search_text": rec.get("search_text", ""),
                    "source_name": rec.get("source_name", ""),
                    "source_type": rec.get("source_type", ""),
                    "dataset_slug": rec.get("dataset_slug", ""),
                    "review_status": rec.get("review_status", "prototype_unverified"),
                    # Disease fields
                    "condition_name": rec.get("condition_name", ""),
                    "symptoms": rec.get("symptoms", []),
                    "description": rec.get("description", ""),
                    "precautions": rec.get("precautions", []),
                    # Medicine fields
                    "medicine_name": rec.get("medicine_name", ""),
                    "generic_name": rec.get("generic_name", ""),
                    "aliases": rec.get("aliases", []),
                    "category": rec.get("category", ""),
                    "uses": rec.get("uses", []),
                    "side_effects": rec.get("side_effects", []),
                    "warnings": rec.get("warnings", []),
                    "salt_composition": rec.get("salt_composition", ""),
                    "mechanism_of_action": rec.get("mechanism_of_action", ""),
                }
            )
            points.append(point)

        # Upload in batches
        try:
            for start in range(0, len(points), batch_size):
                batch = points[start: start + batch_size]
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=True
                )
        except Exception as e:
            raise DatabaseConnectionException(f"Upsert failed: {e}")

    # Legacy upload_points (kept for backward compat)
    def upload_points(self, collection_name: str, chunks: List[dict], dense_vectors: List[list], sparse_vectors: List[dict]):
        if not self.client:
            raise DatabaseConnectionException()
        points = []
        for i, chunk in enumerate(chunks):
            points.append(qmodels.PointStruct(
                id=chunk["chunk_id"],
                vector={
                    "dense": dense_vectors[i],
                    "sparse": qmodels.SparseVector(
                        indices=sparse_vectors[i]["indices"],
                        values=sparse_vectors[i]["values"]
                    )
                },
                payload=chunk
            ))
        try:
            self.client.upload_points(collection_name=collection_name, points=points, wait=True)
        except Exception as e:
            raise DatabaseConnectionException(str(e))

    # ------------------------------------------------------------------
    # Hybrid search — record-type filtered
    # ------------------------------------------------------------------

    def hybrid_search_by_type(
        self,
        collection_name: str,
        query_dense: List[float],
        query_sparse: Dict,
        record_type: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid (dense + sparse) search filtered by record_type.
        Uses client-side RRF fusion for maximum compatibility.
        """
        if not self.client:
            raise DatabaseConnectionException("Qdrant client not connected.")

        record_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="record_type",
                    match=qmodels.MatchValue(value=record_type)
                )
            ]
        )

        try:
            # Dense search

            dense_res = self.client.search(
                collection_name=collection_name,
                query_vector=("dense", query_dense),
                query_filter=record_filter,
                limit=settings.DENSE_TOP_K,
                with_payload=True,
            )

            # Sparse search
            sparse_vec_obj = qmodels.SparseVector(
                indices=query_sparse.get("indices", []),
                values=query_sparse.get("values", [])
            )
            sparse_res = self.client.search(
                collection_name=collection_name,
                query_vector=qmodels.NamedSparseVector(name="sparse", vector=sparse_vec_obj),
                query_filter=record_filter,
                limit=settings.SPARSE_TOP_K,
                with_payload=True,
            )


            # Client-side RRF
            rrf_k = settings.RRF_K
            rrf_scores: Dict[str, float] = {}
            payloads: Dict[str, Any] = {}
            dense_scores_map: Dict[str, float] = {}

            for rank, pt in enumerate(dense_res):
                pid = str(pt.id)
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (rrf_k + rank + 1)
                payloads[pid] = pt.payload
                dense_scores_map[pid] = pt.score

            for rank, pt in enumerate(sparse_res):
                pid = str(pt.id)
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (rrf_k + rank + 1)
                if pid not in payloads:
                    payloads[pid] = pt.payload

            sorted_pids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

            output = []
            for pid in sorted_pids[:limit]:
                output.append({
                    "id": pid,
                    "rrf_score": round(rrf_scores[pid], 6),
                    "dense_score": round(dense_scores_map.get(pid, 0.0), 4),
                    "payload": payloads[pid],
                })
            return output

        except Exception as e:
            raise DatabaseConnectionException(f"Hybrid search failed: {e}")

    # Legacy hybrid search (kept for backward compat with book-based retrieval)
    def search_hybrid(
        self,
        collection_name: str,
        query_dense: list,
        query_sparse: dict,
        limit: int = 4,
        age_group: str = None,
        topic_group: str = None,
    ) -> list:
        if not self.client:
            raise DatabaseConnectionException()
        filter_conds = []
        if age_group:
            filter_conds.append(qmodels.FieldCondition(key="age_group", match=qmodels.MatchValue(value=age_group)))
        if age_group == "adult":
            filter_conds.append(qmodels.FieldCondition(
                key="topic_group",
                match=qmodels.MatchAny(any=["adult_core", "adult_extended"])
            ))
        elif age_group == "child":
            filter_conds.append(qmodels.FieldCondition(
                key="topic_group",
                match=qmodels.MatchValue(value="pediatric")
            ))
        elif topic_group:
            filter_conds.append(qmodels.FieldCondition(key="topic_group", match=qmodels.MatchValue(value=topic_group)))
        query_filter = qmodels.Filter(must=filter_conds) if filter_conds else None

        try:
            dense_res = self.client.search(collection_name=collection_name, query_vector=("dense", query_dense), query_filter=query_filter, limit=18)
            sparse_vector_obj = qmodels.SparseVector(indices=query_sparse["indices"], values=query_sparse["values"])
            sparse_res = self.client.search(collection_name=collection_name, query_vector=qmodels.NamedSparseVector(name="sparse", vector=sparse_vector_obj), query_filter=query_filter, limit=18)

            rrf_scores: Dict[str, float] = {}
            payloads: Dict[str, Any] = {}
            for rank, pt in enumerate(dense_res):
                pid = str(pt.id)
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (60.0 + rank + 1))
                payloads[pid] = pt.payload
            for rank, pt in enumerate(sparse_res):
                pid = str(pt.id)
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (60.0 + rank + 1))
                payloads[pid] = pt.payload

            sorted_pids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
            output = []
            for pid in sorted_pids[:limit]:
                output.append({"id": pid, "score": rrf_scores[pid], "payload": payloads[pid]})
            return output
        except Exception as e:
            raise DatabaseConnectionException(str(e))
