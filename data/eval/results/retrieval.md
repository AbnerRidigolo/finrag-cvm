Banco vetorial: chroma | embeddings: openai (text-embedding-3-small) | reranker: cross-encoder | 43 perguntas | k=5

| Configuração | Hit@1 | Hit@5 | MRR@5 | p50 (ms) | p95 (ms) |
| :-- | --: | --: | --: | --: | --: |
| Denso | 0.65 [0.51–0.79] | 0.88 [0.77–0.98] | 0.75 [0.63–0.85] | 336 | 814 |
| Esparso (BM25) | 0.60 [0.47–0.74] | 0.84 [0.72–0.93] | 0.70 [0.58–0.81] | 3 | 4 |
| Híbrido (RRF) | 0.67 [0.53–0.81] | 0.88 [0.77–0.98] | 0.77 [0.65–0.87] | 320 | 428 |
| Híbrido + re-ranking | 0.81 [0.67–0.91] | 0.95 [0.88–1.00] | 0.87 [0.77–0.94] | 2164 | 2329 |

Diferença pareada (IC 95%, bootstrap com 1000 reamostragens):

| Comparação | Δ Hit@1 | Δ Hit@5 | Δ MRR@5 |
| :-- | --: | --: | --: |
| Híbrido (RRF) − Denso | +0.02 [-0.09 a +0.16] | +0.00 [-0.09 a +0.09] | +0.02 [-0.07 a +0.11] |
| Híbrido (RRF) − Esparso (BM25) | +0.07 [-0.02 a +0.19] | +0.05 [-0.05 a +0.14] | +0.07 [+0.00 a +0.14] |
| Híbrido + re-ranking − Híbrido (RRF) | +0.14 [+0.00 a +0.26] | +0.07 [+0.00 a +0.16] | +0.10 [+0.01 a +0.19] |

Entre colchetes: IC de 95% por bootstrap percentil (semente fixa).
