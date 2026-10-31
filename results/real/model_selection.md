# Model selection by grouped 3-fold cross-validation (train+val only)

| Architecture | Config | Folds | CV AUC (mean ± std) | CV within-resume AUC | Selected |
|---|---|---|---|---|---|
| mlp | mlp_cosine | 3 | 0.6539 ± 0.0211 | 0.6672 | **yes** |
| mlp | mlp_interaction | 3 | 0.6275 ± 0.0613 | 0.6117 |  |
| cnn | cnn_cosine | 3 | 0.6188 ± 0.0542 | 0.6206 | **yes** |
| cnn | cnn_interaction | 3 | 0.5882 ± 0.0265 | 0.5208 |  |
| bilstm | bilstm_cosine | 3 | 0.6599 ± 0.0103 | 0.6579 | **yes** |
| bilstm | bilstm_interaction | 3 | 0.6359 ± 0.0116 | 0.6408 |  |
| sbert | sbert_minilm_cosine | 3 | 0.7781 ± 0.0155 | 0.7064 |  |
| sbert | sbert_minilm_interaction | 3 | 0.7762 ± 0.0323 | 0.7177 |  |
| sbert | sbert_mpnet_interaction | 3 | 0.7857 ± 0.0307 | 0.7313 | **yes** |

Each fold holds out ~93 job descriptions (vs. 42 in the original validation split); the held-out test split is never used for any choice.
