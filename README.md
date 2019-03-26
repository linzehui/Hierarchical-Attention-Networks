# Hierarchical-Attention-Networks

Implementations of Hierarchica Attention Networks in Pytorch.

One hightlight of this implementation of HAN is its efficiency. Unlike most implementations on GitHub, we don't pad empty sentences to make batches, which makes the training process slower. Instead, we use pack_padded_sequence/pad_packed_sequence twice in WordAttenModel and SentAttenModel, respectively. We first select n batches documents and unfold into sentences and process them as a batch to get sentence representations. After that we make the sentence representations as a batch and process them to get the final document representations. 

Without careful hyperparameters search, we get the result of 68+ in Yelp14, while the reported result is 70.5. 
