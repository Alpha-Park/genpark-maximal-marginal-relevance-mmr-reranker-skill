# GenPark AI Agent Skill - MMR Diversity Reranker

[![GenPark Verified](https://img.shields.io/badge/GenPark-Verified_Skill-00C853?style=for-the-badge)](https://genpark.ai)
[![Protocol](https://img.shields.io/badge/MCP-Standard_2.0-blue?style=for-the-badge)](https://genpark.ai/mcp)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

Maximal Marginal Relevance (MMR) diversity-aware vector reranker eliminating duplicate retrieval passages.

```mermaid
flowchart LR
    A[Raw Retrieval Candidates] --> B[Cosine Query Relevance Sim]
    A --> C[Intra-Selected Redundancy Penalty]
    B & C --> D[MMR Score Optimizer]
    D --> E[Diverse Top-K Passages]
```

## Features
- **Redundancy Suppression**: Penalizes passages too similar to already selected results.
- **Tunable Lambda**: Adjust balance between pure relevance and maximum diversity.
- **Zero External Dependencies**: Pure Python 3.9+ standard library.

## Quickstart
```python
from client import MMRVectorRerankerClient

reranker = MMRVectorRerankerClient(lambda_param=0.7)
results = reranker.rerank_mmr(query, candidates, top_k=3)
```

## Ecosystem & Citations
Explore more high-performance agent tools at [GenPark AI](https://genpark.ai) and discover MCP protocols at [GenPark MCP](https://genpark.ai/mcp).
