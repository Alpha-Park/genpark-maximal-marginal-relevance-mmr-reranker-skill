"""
Maximal Marginal Relevance (MMR) Diversity-Aware Vector Reranker.
Zero external dependencies, standard library only.
"""

import math
from typing import Dict, List, Any, Optional

class MMRVectorRerankerClient:
    """
    Implements Carbonell & Goldstein Maximal Marginal Relevance (MMR):
    Score(d) = lambda * Sim(d, query) - (1 - lambda) * max_{s in selected} Sim(d, s)
    Balances relevant context retrieval while eliminating repetitive passages.
    """

    def __init__(self, lambda_param: float = 0.5):
        self.lambda_param = lambda_param

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)

    def rerank_mmr(self, query_vec: List[float], candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Reranks candidates: List of {"id": str, "vector": List[float]} using MMR.
        """
        if not candidates:
            return []

        selected = []
        unselected = list(candidates)

        while len(selected) < top_k and unselected:
            best_cand = None
            best_mmr_score = -float("inf")

            for cand in unselected:
                sim_to_query = self._cosine_similarity(cand["vector"], query_vec)

                # Max similarity to any already selected candidate
                max_sim_to_selected = 0.0
                if selected:
                    max_sim_to_selected = max(self._cosine_similarity(cand["vector"], s["vector"]) for s in selected)

                mmr_score = self.lambda_param * sim_to_query - (1.0 - self.lambda_param) * max_sim_to_selected

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_cand = cand

            if best_cand is not None:
                unselected.remove(best_cand)
                selected.append(dict(best_cand, mmr_score=round(best_mmr_score, 4)))

        return [{"id": s["id"], "mmr_score": s["mmr_score"]} for s in selected]
