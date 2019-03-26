import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from functional import *


class WordAttnModel(nn.Module):
    def __init__(self, embed_dim, word_hidden_dim,
                 rnn_type='GRU',attn_dropout=0.1):
        super(WordAttnModel, self).__init__()
        self.word_hidden_dim=word_hidden_dim
        if rnn_type.upper() not in ['GRU', 'LSTM']:
            raise ValueError("rnn type must be in ['GRU','LSTM'] ")
        self.rnn_type = rnn_type.upper()
        self.word_RNN=getattr(nn, self.rnn_type)\
            (embed_dim,word_hidden_dim,batch_first=True,bidirectional=True)

        # self.W_word=nn.Conv1d(2 * word_hidden_dim, 2 * word_hidden_dim, 1, bias=True)
        self.W_word=nn.Linear(2*word_hidden_dim,2*word_hidden_dim,bias=True)  # test this
        self.u_w = nn.Parameter(torch.Tensor(2 * word_hidden_dim, 1))

        self.attn_drop=nn.Dropout(attn_dropout)
        self._init_paras()

    def forward(self, seq,mask):
        """

        :param seq: batch,seq,dim
        :param mask: batch,seq
        :return:
        """
        mask=mask.to(seq.device)

        # --- get hiden state--- #
        batch_size=seq.size(0)
        packed_seq,sorted_index=pack_seq(seq,mask)
        word_init_state=self.init_hidden(batch_size)
        word_init_state=word_init_state.to(seq.device)
        word_output,_=self.word_RNN(packed_seq,word_init_state) # get packedSequence,batch_first=true
        word_output,_ = pad_packed_sequence(word_output,batch_first=True)  # batch,seq,dim*2
        word_output=word_output.index_copy(0,torch.tensor(sorted_index).to(seq.device),word_output)  # unsort to origin order
        word_output=word_output*mask.float().unsqueeze(-1)


        # ---get u_i --- #
        # u_i=torch.tanh(self.W_word(word_output.transpose(1,2)))  # bz,dim,seq_len
        u_i=torch.tanh(self.W_word(word_output))
        u_i=u_i.transpose(1,2)

        # ---get attention ---#
        batch_size,seq_dim,seq_len=u_i.size()
        _u_i=u_i.transpose(1,2).contiguous().view(-1,seq_dim)  # bz*seq,dim
        attn=torch.mm(_u_i,self.u_w).contiguous().\
            view(batch_size,seq_len,-1).squeeze(-1) #bz,seq_len
        _mask=(~mask).to(u_i.device)  # fill pad with 1
        attn=attn.masked_fill(_mask,-np.inf)
        attn=F.softmax(attn,dim=-1)   # batch,seq_len
        attn=self.attn_drop(attn)   # dropout

        # --- get sentence vectors--- #
        s_i=torch.bmm(u_i,attn.unsqueeze(-1)).squeeze(-1)  # # (bz,dim,seq_len)* (bz,seq_len,1)  --> bz,dim
        return s_i,attn



    def init_hidden(self, batch_size):  # todo add LSTM support or cancel this function
        if self.rnn_type == 'GRU':
            init_state = torch.zeros(( 2,batch_size, self.word_hidden_dim))
        else:
            pass
        return init_state


    def _init_paras(self):
        # nn.init.xavier_normal_(self.W_word.weight)
        nn.init.xavier_normal_(self.u_w)
        if self.rnn_type == 'GRU':
            self._gru_init()
        else:
            pass  # TODO add LSTM support

    def _gru_init(self):
        nn.init.orthogonal_(self.word_RNN.weight_ih_l0.data)
        nn.init.orthogonal_(self.word_RNN.weight_ih_l0_reverse.data)
        nn.init.orthogonal_(self.word_RNN.weight_hh_l0.data)
        nn.init.orthogonal_(self.word_RNN.weight_hh_l0_reverse.data)

        # bias
        self.word_RNN.bias_ih_l0.data.zero_()
        self.word_RNN.bias_ih_l0_reverse.data.zero_()
        self.word_RNN.bias_hh_l0.data.zero_()
        self.word_RNN.bias_hh_l0_reverse.data.zero_()















