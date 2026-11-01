# copernicus.embeddings

Encoders turn texts into chunked, pooled, unit-length vectors. An embedding set holds the vectors with a record of the encoder that made them. Importing this module does not import torch; the encoder does when it loads a model, and the `embed` extra installs it.

::: copernicus.embeddings
