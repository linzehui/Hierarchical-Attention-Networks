import torch.nn as nn
import torch
import torch.nn.functional as F
from functional import *



class SentAttnModel(nn.Module):
    def __init__(self, sent_vec_dim=100, sent_hidden_dim=50,
                 rnn_type='GRU',attn_dropout=0.1):
        super(SentAttnModel, self).__init__()
        self.sent_hidden_dim = sent_hidden_dim
        self.sent_vec_dim = sent_vec_dim
        if rnn_type.upper() not in ['GRU', 'LSTM']:
            raise ValueError("rnn type must be in ['GRU','LSTM'] ")
        self.rnn_type = rnn_type.upper()

        self.sent_RNN=getattr(nn,self.rnn_type)\
            (sent_vec_dim,sent_hidden_dim,batch_first=True,bidirectional=True)

        # self.W_sent=nn.Conv1d(2 * sent_hidden_dim, 2 * sent_hidden_dim, 1, bias=True)
        self.W_sent=nn.Linear(2*sent_hidden_dim,2*sent_hidden_dim,bias=True)
        self.u_s = nn.Parameter(torch.Tensor(2 * sent_hidden_dim, 1))
        self.attn_drop=nn.Dropout(attn_dropout)

        self._init_paras()

    def forward(self, seq,mask):
        """

        :param seq: bz,max_n_sents,dim
        :param mask: bz,max_n_sents
        :return: doc_vec:bz,dim;  attn:batch,seq_len
        """

        mask=mask.to(seq.device)

        # --- get hidden state --- #
        batch_size=seq.size(0)
        packed_seq,sorted_index=pack_seq(seq,mask)
        sent_init_state=self.init_hidden(batch_size)
        sent_init_state=sent_init_state.to(seq.device)
        sent_output,_=self.sent_RNN(packed_seq,sent_init_state)  # get packedSequence,batch_first=true
        sent_output,_=pad_packed_sequence(sent_output,batch_first=True)
        sent_output=sent_output.index_copy(0,torch.tensor(sorted_index).to(seq.device),sent_output)   # unsort
        sent_output=sent_output*mask.float().unsqueeze(-1)

        # --- get u_i --- #
        # u_i=torch.tanh(self.W_sent(sent_output.transpose(1,2)))  # bz,seq_len,dim
        u_i=torch.tanh(self.W_sent(sent_output))
        u_i=u_i.transpose(1,2)

        # --- get attention --- #
        batch_size,seq_dim,seq_len=u_i.size()
        _u_i=u_i.transpose(1,2).contiguous().view(-1,seq_dim)  # bz*seq,dim
        attn=torch.mm(_u_i,self.u_s).contiguous().\
            view(batch_size,seq_len,-1).squeeze(-1) #bz,seq_len
        _mask=(~mask.byte()).to(u_i.device)  # fill pad with 1
        attn=attn.masked_fill(_mask,-np.inf)
        attn=F.softmax(attn,dim=-1)   # batch,seq_len
        attn=self.attn_drop(attn)   # dropout

        # --- get doc vectors --- #
        v = torch.bmm(u_i,attn.unsqueeze(-1)).squeeze(-1)   # batch,dim

        return v,attn



    def init_hidden(self, batch_size):  # todo add LSTM support or cancel this function
        if self.rnn_type == 'GRU':
            init_state = torch.zeros((2, batch_size, self.sent_hidden_dim))
        else:
            pass
        return init_state


    def _init_paras(self):
        # nn.init.xavier_normal_(self.W_sent.weight)
        nn.init.xavier_normal_(self.u_s)
        if self.rnn_type == 'GRU':
            self._gru_init()
        else:
            pass  # TODO add LSTM support

    def _gru_init(self):
        nn.init.orthogonal_(self.sent_RNN.weight_ih_l0.data)
        nn.init.orthogonal_(self.sent_RNN.weight_ih_l0_reverse.data)
        nn.init.orthogonal_(self.sent_RNN.weight_hh_l0.data)
        nn.init.orthogonal_(self.sent_RNN.weight_hh_l0_reverse.data)

        # bias
        self.sent_RNN.bias_ih_l0.data.zero_()
        self.sent_RNN.bias_ih_l0_reverse.data.zero_()
        self.sent_RNN.bias_hh_l0.data.zero_()
        self.sent_RNN.bias_hh_l0_reverse.data.zero_()




