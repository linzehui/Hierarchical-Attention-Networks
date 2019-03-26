import torch
import torch.nn as nn
from config import ops

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim,
                 pretrained_matrix=None):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_matrix is not None:
            pretrained_matrix=torch.from_numpy(pretrained_matrix).type(torch.FloatTensor)
            self.embedding.weight= nn.Parameter(pretrained_matrix,
                                                requires_grad=not ops.freeze)

    def forward(self, seq):
        """
        :param seq: batch,max_sent_len
        :return:
        """
        return self.embedding(seq)

