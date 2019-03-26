import torch.nn as nn
import torch
import torch.nn.functional as F
from model.EmbeddingLayer import EmbeddingLayer
from model.SentAttnModel import SentAttnModel
from model.WordAttnModel import WordAttnModel
from functional import *


class HierAttnModel(nn.Module):
    def __init__(self, word_hidden_dim, sent_hidden_dim,
                 vocab_size, embed_dim,
                 n_label,
                 rnn_type='GRU',
                 pretrained_matrix=None,
                 attn_dropout=0.1):
        super(HierAttnModel, self).__init__()
        self.word_hidden_dim = word_hidden_dim
        self.sent_hidden_dim = sent_hidden_dim
        self.n_label = n_label
        self.rnn_type = rnn_type
        self.vocab_size = vocab_size
        sent_vec_dim = 2 * word_hidden_dim

        self.embedding = EmbeddingLayer(vocab_size, embed_dim, pretrained_matrix)
        self.word_attn_model=WordAttnModel(
            embed_dim,word_hidden_dim,rnn_type,attn_dropout
        )
        self.sent_attn_model=SentAttnModel(
            sent_vec_dim,sent_hidden_dim,rnn_type,attn_dropout
        )

        self.fc=nn.Linear(2*sent_hidden_dim,n_label)

        self._init_paras()

    def forward(self, seq,doc_n_sents):

        sent_mask=get_non_pad_mask(seq)
        x=self.embedding(seq)

        s_i,sent_attn=self.word_attn_model(x,sent_mask)
        batch_sent_vec,doc_mask=batchify_sent_vec(s_i,doc_n_sents)

        v,doc_attn=self.sent_attn_model(batch_sent_vec,doc_mask)  # batch,dim; batch,seq_len

        logics=self.fc(v)

        return logics,sent_attn,doc_attn


    def _init_paras(self):
        nn.init.xavier_normal_(self.fc.weight)


