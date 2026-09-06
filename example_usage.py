"""
Demonstration of genpark-maximal-marginal-relevance-mmr-reranker-skill
"""

from client import MMRVectorRerankerClient

def main():
    reranker = MMRVectorRerankerClient(lambda_param=0.6)

    query = [1.0, 0.0, 0.0]

    # Candidate vectors: doc1 and doc2 are almost duplicates
    candidates = [
        {"id": "doc1", "vector": [0.99, 0.02, 0.01]},
        {"id": "doc2_near_duplicate", "vector": [0.98, 0.03, 0.01]},
        {"id": "doc3_diverse_perspective", "vector": [0.85, 0.35, 0.05]}
    ]

    reranked = reranker.rerank_mmr(query, candidates, top_k=2)
    print("=== MMR DIVERSITY RERANKING REPORT ===")
    for r in reranked:
        print(f"[{r['id']}] MMR Score: {r['mmr_score']}")

if __name__ == "__main__":
    main()
