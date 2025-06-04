| Model                |   accuracy |       f1 | Epoch   | Notes                                       |
|:---------------------|-----------:|---------:|:--------|:--------------------------------------------|
| IndoBERT-Lite        |   0.97     | 0.983413 | 2       | indobert-lite-base-p1                       |
| IndoBERTweet         |   0.957667 | 0.97671  | 2       | indobertweet-base-uncased                   |
| BERT                 |   0.972    | 0.984542 | 2       | indobert-base-p1                            |
| DistilBERT           |   0.903    | 0.949028 | 2       | indobertweet-uncased                        |
| Ensemble BERT+Distil |   0.969    | 0.982995 | -       | Average logits BERT & DistilBERT            |
| Ensemble IndoBERT    |   0.97     | 0.983492 | -       | Average logits IndoBERT-Lite & IndoBERTweet |