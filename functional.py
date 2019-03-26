import torch
from preprocessing import *
from sampler import *
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataset import *




def pack_seq(seq, mask):
    """
    pack seq as PackedSequence
    :param seq: batch,seq,dim
    :param mask: batch,seq
    :return: PackedSequence,sorted_index
    """
    lengths = np.array(mask.sum(-1).long().tolist())
    sorted_index = np.argsort(-lengths)  # get index
    sorted_lengths = lengths[sorted_index]  # descend order
    sorted_index = torch.Tensor(sorted_index).long().to(seq.device)
    sorted_seq = seq.index_select(0, sorted_index)
    packed_seq = pack_padded_sequence(sorted_seq, sorted_lengths,batch_first=True)

    return packed_seq, sorted_index.cpu().numpy()


def collate_fn(batch):
    docs, labels = [], []
    for b in batch:
        docs.append(b[0])
        labels.append(b[1])

    doc_n_sents = [len(doc) for doc in docs]  # record the number of sentences in every document
    if all(doc_n_sents) is False:  # test code
        raise ValueError('Find doc with 0 length')

    max_sent_len = max(len(sentence) for doc in docs for sentence in doc)
    sentences = [sent for doc in docs for sent in doc]  # flatten

    batch_seq = np.array([
        sent + [0] * (max_sent_len - len(sent))
        for sent in sentences
    ])  # pad with 0

    batch_label = np.array(labels)

    batch_seq = torch.LongTensor(batch_seq)
    batch_label = torch.LongTensor(batch_label)

    return batch_seq, doc_n_sents, batch_label


def get_non_pad_mask(seq):
    """fill non pad with 1 """
    assert seq.dim() == 2
    return seq.ne(0)  # bz,seq  type:uint8


def batchify_sent_vec(sent_vec, doc_n_sents):
    """
    :param:
        sent_vec: bz,dim;  this batch_size does not represent the number of docs; bz here equal n_doc*average_len in document
        doc_n_sents : list containing number of sentences in documents

    re-batchify sentences vectors
    we have to re-batchify because we mixed sentences among documents

    :return:
        batch_sent_vec: sentence_bz,max_n_sents,dim
        mask: bz,max_n_sents  non pad with 1

    """
    max_n_sents = max(doc_n_sents)
    vec_dim = sent_vec.size(1)
    n_docs = len(doc_n_sents)  # number of documents,the 'real' batch_size

    # --- get batched_vec--- #

    vec_chunks = torch.split(sent_vec, doc_n_sents)  # tuple
    batch_sent_vec = torch.tensor([]).to(sent_vec.device)  # empty
    for c in vec_chunks:  # c: n,dim
        n_c = c.size(0)  # number of c
        padded_zeros = torch.zeros(max_n_sents - n_c, vec_dim).to(c.device)  # all zero
        padded_c = torch.cat((c, padded_zeros), dim=0)  # max_n_sents,dim;  with padding
        padded_c = padded_c.unsqueeze(0)  # 1, max_n_sents,dim
        batch_sent_vec = torch.cat((batch_sent_vec, padded_c), dim=0)

    # --- get mask --- #
    mask = torch.zeros(n_docs, max_n_sents)
    for i, n_sent in enumerate(doc_n_sents):
        mask[i, :n_sent] = torch.ones(1, n_sent)

    return batch_sent_vec, mask.byte()


def save_chkpt(model_state_dict, epoch, valid_accu):  # todo learn to write this  put it to utils

    checkpoint = {
        'model': model_state_dict,
        'epoch': epoch,
        'valid_accu': valid_accu
    }
    chkpt_name = 'accu{accu:3.3f}.chkpt'.format(accu=100 * valid_accu) + 'epoch' + str(epoch)
    print('[Info] Saving Model...')
    torch.save(checkpoint, chkpt_name)


def get_dataset(train_file,valid_file,test_file,ops):
    max_sent_len = ops.max_sent_len
    max_num_sent = ops.max_num_sent
    min_freq = ops.min_freq
    min_num_sent = ops.min_num_sent

    train_docs, train_labels = read_docs_from_file(train_file, max_sent_len, max_num_sent, min_num_sent)
    valid_docs, valid_labels = read_docs_from_file(valid_file, max_sent_len)
    test_docs, test_labels = read_docs_from_file(test_file, max_sent_len)
    vocab = build_vocab(train_docs+valid_docs+test_docs, min_freq)   # use train+valid docs to build

    train_docs_indexed = indexed(train_docs, vocab)
    valid_docs_indexed = indexed(valid_docs, vocab)
    test_docs_indexed = indexed(test_docs, vocab)

    train_docs_dataset = DocsDataset(train_docs_indexed, train_labels)
    valid_docs_dataset = DocsDataset(valid_docs_indexed, valid_labels)
    test_docs_dataset = DocsDataset(test_docs_indexed, test_labels)

    train_docs_dataset.shuffle()

    return train_docs_dataset,valid_docs_dataset,test_docs_dataset, train_docs,valid_docs,test_docs, vocab


def get_dataloader(train_dataset, valid_dataset,test_dataset, ops):

    batch_size = ops.batch_size

    train_docs_indexed=train_dataset.documents

    if ops.sampler == 'MaxSentence':
        shuffle = False
        sampler = MaxSentenceSampler(train_docs_indexed,
                                     reverse=ops.reverse)
    elif ops.sampler == 'Random':
        shuffle = True
        sampler = None

    elif ops.sampler == 'NumSentence':
        shuffle = False
        sampler = NumSentenceSampler(train_docs_indexed,
                                     reverse=ops.reverse)


    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              sampler=sampler,
                              shuffle=shuffle,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              collate_fn=collate_fn)

    if ops.sampler == 'NumSentence':
        bucketwrapper=DataLoaderBucketWrapper(dataloader=train_loader)
        return bucketwrapper, valid_loader ,test_loader

    return train_loader,valid_loader

def shuffle_dataloader(dataloader,ops):   # deprecated
    batch_size=ops.batch_size
    docs_dataset=dataloader.dataset
    docs_dataset.shuffle()
    docs_dataset_documents=docs_dataset.documents
    if ops.sampler == 'MaxSentence':
        shuffle = False
        sampler = MaxSentenceSampler(docs_dataset_documents,
                                     reverse=ops.reverse)
    elif ops.sampler == 'Random':
        shuffle = True
        sampler = None

    elif ops.sampler == 'NumSentence':
        shuffle = False
        sampler = NumSentenceSampler(docs_dataset_documents,
                                     reverse=ops.reverse)
    dataloader=DataLoader(dataset=docs_dataset,
                          batch_size=batch_size,
                          sampler=sampler,
                          shuffle=shuffle,
                          collate_fn=collate_fn)

    return dataloader


def construct_embedding_matrix_from_file(file_name, vocab, embed_dim=200):
    from gensim.models import KeyedVectors
    from gensim.test.utils import datapath, get_tmpfile
    from gensim.scripts.glove2word2vec import glove2word2vec
    glove_file = datapath(file_name)
    word2vec_file = get_tmpfile('word2vec.txt')
    glove2word2vec(glove_file, word2vec_file)
    model = KeyedVectors.load_word2vec_format(word2vec_file)

    vocab_size = len(vocab)
    embedding_matrix = np.zeros(shape=(vocab_size, embed_dim))
    words = model.vocab

    n_init = 0
    for i in range(vocab_size):
        word = vocab.get_word(i)
        if word in words:
            embedding_matrix[i][:] = model[word]
            n_init += 1
        else:
            continue

    print('[Info] Pretrained Embedding Constructed from file,with {n_init} initialized,'
          'The whole vocab size is {vocab_size}'.format(n_init=n_init, vocab_size=len(vocab)))
    return embedding_matrix

def construct_embedding_matrix_from_model(w2v_model,vocab,embed_dim=200):
    vocab_size=len(vocab)
    words=list(w2v_model.wv.vocab.keys())
    embedding_matrix=np.zeros(shape=(vocab_size,embed_dim))

    n_init=0
    for i in range(1,vocab_size):
        word=vocab.get_word(i)
        if word in words:
            embedding_matrix[i][:]=w2v_model.wv[word]
            n_init+=1
        else:
            embedding_matrix[i][:]=np.random.randn(embed_dim)

    print('[Info] Pretrained Embedding Constructed from model,with {n_init} initialized,'
          'The whole vocab size is {vocab_size}'.format(n_init=n_init, vocab_size=len(vocab)))

    return embedding_matrix


def train_word2vec(texts,file_name,dim=200,min_count=5):
    from gensim.models import Word2Vec
    word2vec = Word2Vec(texts, size=dim, min_count=min_count)
    word2vec.save(file_name)
    print('[Info] Word2Vec Model Trained')
    return word2vec

