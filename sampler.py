from random import shuffle
from torch.utils.data import Sampler




class MaxSentenceSampler(Sampler):    # 根据一个doc的中最大长度的句子来排的
    def __init__(self,data_source,reverse=True):
        self.data_source=data_source
        self.reverse=reverse  # from max to min

    def __iter__(self):
        max_sent_lens=[max([len(sent) for sent in sentences])   # max_sent_len in docs
                       for sentences in self.data_source]

        sorted_index=[i[0] for i in sorted(enumerate(max_sent_lens),
                                           key=lambda x:x[1],
                                           reverse=self.reverse)]   # get sorted index
        return iter(sorted_index)

    def __len__(self):
        return len(self.data_source)


class NumSentenceSampler(Sampler):     # 根据一个doc所拥有的句子的长度来排的
    def __init__(self,data_source,reverse=True):
        self.data_source=data_source
        self.reverse=reverse

    def __iter__(self):
        n_sentences=[len(sentences) for sentences in self.data_source]
        sorted_index=[i[0] for i in sorted(enumerate(n_sentences),
                                           key=lambda x:x[1],
                                           reverse=self.reverse)]

        return iter(sorted_index)

    def __len__(self):
        return len(self.data_source)


