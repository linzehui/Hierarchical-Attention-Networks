from torch.utils.data import Dataset
import random
from torch.utils.data import DataLoader

class Vocabulary():

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.__vocab_size = 0
        self.add_word('<pad>')
        self.add_word('<UNK>')

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = self.__vocab_size
            self.__vocab_size += 1

    def __len__(self):
        return self.__vocab_size

    def get_index(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx['<UNK>']

    def get_word(self, idx):
        return self.idx2word[idx]


class DocsDataset(Dataset):
    def __init__(self, documents, labels):
        self.documents = documents  # documents are tokenized and indexed
        self.labels = labels

    def __len__(self):
        assert len(self.documents) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        return self.documents[idx], self.labels[idx]

    def shuffle(self):
        start_state=random.getstate()
        random.shuffle(self.documents)
        random.setstate(start_state)
        random.shuffle(self.labels)

class DataLoaderBucketWrapper():   # for training shuffle
    def __init__(self,dataloader):
        self.dataloader=dataloader
        self.all_batches=self._get_all_batches(dataloader)

    def _get_all_batches(self,dataloader):
        all_batches=[]
        for batch in dataloader:
            all_batches.append(batch)

        return all_batches

    def __iter__(self):   # tested
        import random
        n_batch=len(self.all_batches)
        shuffle_index=list(range(n_batch))
        random.shuffle(shuffle_index)
        self.shuffle_index= iter(shuffle_index)  # iter wrapper
        return self

    def __len__(self):
        return len(self.dataloader)

    def __next__(self):
        try:
            index=next(self.shuffle_index)
            return self.all_batches[index]
        except StopIteration:
            self.shuffle_dataloader()
            raise StopIteration

    def shuffle_dataloader(self):
        batch_size=self.dataloader.batch_size
        docs_dataset=self.dataloader.dataset
        docs_dataset.shuffle()
        docs_dataset_documents=docs_dataset.documents  # after shuffle
        sampler=self.dataloader.sampler
        reverse=sampler.reverse
        collate_fn=self.dataloader.collate_fn
        new_sampler=sampler.__class__(docs_dataset_documents,
                                                  reverse=reverse)
        self.dataloader=DataLoader(dataset=docs_dataset,
                                   batch_size=batch_size,
                                   sampler=new_sampler,
                                   collate_fn=collate_fn)

        self.all_batches=self._get_all_batches(self.dataloader)











