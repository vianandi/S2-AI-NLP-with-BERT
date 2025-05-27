| Model      |   Size_MB |   Parameters_M |   Avg_Inference_s | Notes                              |   Throughput_samples_per_second |
|:-----------|----------:|---------------:|------------------:|:-----------------------------------|--------------------------------:|
| BERT       |    497.8  |         124.44 |            0.0401 | indobert-base-p1                   |                           24.94 |
| DistilBERT |    442.26 |         110.56 |            0.0405 | indobertweet-base-uncased          |                           24.69 |
| Ensemble   |    940.05 |         235    |            0.0794 | Average logits (BERT + DistilBERT) |                           12.59 |