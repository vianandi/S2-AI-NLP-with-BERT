| Model      |   Size_MB |   Parameters_M |   Avg_Inference_s | Notes                              |   Throughput_samples_per_second |
|:-----------|----------:|---------------:|------------------:|:-----------------------------------|--------------------------------:|
| BERT       |    497.8  |         124.44 |            0.0334 | indobert-base-p1                   |                           29.94 |
| DistilBERT |    442.26 |         110.56 |            0.0331 | indobertweet-base-uncased          |                           30.21 |
| Ensemble   |    940.06 |         235    |            0.0673 | Average logits (BERT + DistilBERT) |                           14.86 |