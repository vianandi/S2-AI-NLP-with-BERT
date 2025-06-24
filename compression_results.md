| Model      |   Size_MB |   Parameters_M |   Avg_Inference_s | Notes                              |   Throughput_samples_per_second |
|:-----------|----------:|---------------:|------------------:|:-----------------------------------|--------------------------------:|
| BERT       |    497.8  |         124.44 |            0.0322 | indobert-base-p1                   |                           31.06 |
| DistilBERT |    442.26 |         110.56 |            0.0309 | indobertweet-base-uncased          |                           32.36 |
| Ensemble   |    940.06 |         235    |            0.0614 | Average logits (BERT + DistilBERT) |                           16.29 |