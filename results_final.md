| Model                |   Accuracy |       F1 |   Size_MB |   Params_M |   Inference_s |   Throughput | Notes                                       |
|:---------------------|-----------:|---------:|----------:|-----------:|--------------:|-------------:|:--------------------------------------------|
| IndoBERT-Lite        |   0.97     | 0.983413 |    250.32 |      66    |        0.0287 |        34.84 | indobert-lite-base-p1                       |
| IndoBERTweet         |   0.957667 | 0.97671  |    410.22 |     110    |        0.035  |        28.57 | indobertweet-base-uncased                   |
| BERT                 |   0.972    | 0.984542 |    497.8  |     124.44 |        0.0401 |        24.94 | indobert-base-p1                            |
| DistilBERT           |   0.903    | 0.949028 |    442.26 |     110.56 |        0.0405 |        24.69 | indobertweet-uncased                        |
| Ensemble BERT+Distil |   0.969    | 0.982995 |    940.05 |     235    |        0.0794 |        12.59 | Average logits BERT & DistilBERT            |
| Ensemble IndoBERT    |   0.97     | 0.983492 |    660.54 |     176    |        0.0642 |        15.58 | Average logits IndoBERT-Lite & IndoBERTweet |