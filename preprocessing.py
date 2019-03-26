from dataset import Vocabulary
from collections import Counter
from utils import *


# def read_docs_from_file(file,max_sent_len=10000,max_num_sent=10000,min_num_sent=1,keep_case=False):
#     """
#
#     :param file: this file should have been preprocessed
#     :param max_sent_len:
#     :param max_num_sent:  max number of sentences in a document
#     :param keep_case:
#     :param min_num_sent: get rid of small doc
#     :return:  documents:list(doc[sent['a','b'],sent[]] , [[],[]]); labels : list of numbers
#     """
#     documents=[]
#     labels=[]
#     with open(file,'r') as f:
#         for line in f:
#             l=line.split('\t\t')
#             label,document=l[2],l[3]
#             if not document:   # document is empty
#                 continue
#             if not keep_case:
#                 document=document.lower()
#
#             sentences=document.split('.')
#             if len(sentences)<min_num_sent:     # get rid of small doc
#                 continue
#             if len(sentences)>max_num_sent:
#                 sentences=sentences[:max_num_sent]
#
#             collection = []       # sentence collection
#             for sent in sentences:
#                 s=clean_str(sent).split()[:max_sent_len]
#                 # s=clean_stopwords(s)
#                 if s:     # not empty
#                     s = stem(s)
#                     collection.append(s)
#             # sentences=[ clean_stopwords(clean_str(sent).split()[:max_sent_len])  for sent in sentences
#             #            if sent.split() ]
#             if not collection:    # sentences is empty
#                 continue
#             documents.append(collection)
#             labels.append(int(label)-1)  # start from 0
#
#
#     print('[Info] Finished reading from file',
#           'with {num} pieces of data'.format(num=len(labels)))
#     return documents,labels


def read_docs_from_file(file, max_sent_len=10000, max_num_sent=10000, min_num_sent=1, keep_case=False):
    """
    # 测试行为正常，TODO 是否需要做clean
    :param file: this file should have been preprocessed
    :param max_sent_len:
    :param max_num_sent:  max number of sentences in a document
    :param keep_case:
    :param min_num_sent: get rid of small doc
    :return:  documents:list(doc[sent['a','b'],sent[]] , [[],[]]); labels : list of numbers
    """
    documents = []
    labels = []
    with open(file, 'r') as f:
        for line in f:
            l = line.split('\t\t')
            document, label = l[3], l[2]
            if not document:  # document is empty
                continue
            if not keep_case:
                document = document.lower()

            sentences = document.split('.')
            if len(sentences) < min_num_sent:  # get rid of small doc
                continue
            if len(sentences) > max_num_sent:
                sentences = sentences[:max_num_sent]

            # sentences = [sent.split()[:max_sent_len] for sent in sentences
            #              if sent.split()]
            sentences = [clean_str(sent)[:max_sent_len] for sent in sentences
                           if clean_str(sent)]
            if not sentences:  # empty
                continue
            documents.append(sentences)
            labels.append(int(label) - 1)  # start from 0

    print('[Info] Finished reading from file',
          'with {num} pieces of data'.format(num=len(labels)))
    return documents, labels


def build_vocab(documents, min_word_cnt=5):
    vocab = Vocabulary()

    all_words = [word for sentences in documents
                 for sent in sentences
                 for word in sent]
    full_vocab = Counter(all_words).most_common()
    print('[Info] Original Vocabulary size =', len(full_vocab))

    for item in full_vocab:
        if item[1] > min_word_cnt:
            vocab.add_word(item[0])
        else:
            break

    print('[Info] Trimmed vocabulary size = {},'.format(len(vocab)),
          'each with minimum occurrence = {}'.format(min_word_cnt))
    print("[Info] Ignored word count = {}".format(len(full_vocab) - len(vocab)))

    return vocab


def indexed(documents, vocab):
    return [[[vocab.get_index(word) for word in sent]
             for sent in sentences]
            for sentences in documents]


# def preprocess(file):
#     documents = []
#     labels = []
#     cnt = 0
#     with open(file, 'r') as f:
#         for line in f:
#             l = line.split('\t\t')
#             label, document = l[2], l[3]
#             print(document)
#             document = clean_str(document)
#             # document=clean_stopwords(document.split())
#             # document=stem(document)
#             print(document)
#             if document:
#                 # print('[Info] Finished processing {num} pieces of data'.format(num=cnt))
#                 cnt += 1
#                 documents.append(document)
#                 labels.append(label)
#
#     file_name = file.split('.')[0]
#     with open('cleaned_' + file_name, 'w') as outfile:
#         for i in range(len(labels)):
#             outfile.write(" ".join(documents[i]))
#             outfile.write('\t\t')
#             outfile.write(labels[i])
#             outfile.write('\n')
#
#     print('[Info] Finished proprecessing')



if __name__ == '__main__':
    import random
    def flat(doc):
        doc=[word for sent in doc for word in sent]
        return " ".join(doc)
    docs,labels=read_docs_from_file('yelp-2013-train.txt.ss')
    counter_label=Counter(labels)
    print(counter_label)
    # start_state = random.getstate()
    # random.shuffle(docs)
    # random.setstate(start_state)
    # random.shuffle(labels)
    # with open('yelp14-dev-preprocessed','w') as f:
    #     for doc,label in zip(docs,labels):
    #         f.write(str(label))
    #         f.write('\t')
    #         f.write(flat(doc))
    #         f.write('\n')
    #
    # print('Finish processing')

