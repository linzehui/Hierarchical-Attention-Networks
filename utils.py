import re


# def clean_str(string):
#     # from bs4 import BeautifulSoup
#     # string = BeautifulSoup(string, "lxml").text   # borrow from github  can be very slow
#     string = re.sub("<sssss>","",string)
#     string = re.sub("-lrb-","",string)
#     string = re.sub("-rrb-","",string)
#     # string = re.sub(r"[^A-Za-z0-9(),!?\'\`\.]", " ", string)
#     # # string = re.sub(r"[0-9]+\.*[0-9]*", " <NUM> ", string)
#     # string = re.sub(r'\d+\.\d+|\d{2,}', "<NUM>", string)
#     # string = re.sub(r"\'s", " \'s", string)
#     # string = re.sub(r"\'ve", " \'ve", string)
#     # string = re.sub(r"n\'t", " n\'t", string)
#     # string = re.sub(r"\'re", " \'re", string)
#     # string = re.sub(r"\'d", " \'d", string)
#     # string = re.sub(r"\'ll", " \'ll", string)
#     # string = re.sub(r",", " , ", string)
#     # string = re.sub(r"!", " ! ", string)
#     # string = re.sub(r"\(", " ( ", string)
#     # string = re.sub(r"\)", " ) ", string)
#     # string = re.sub(r"\?", " ? ", string)
#     # string = re.sub(r"\s{2,}", " ", string)
#
#     return string.strip()

def clean_str(sentence):
    """
    heavily borrowed from github
    https://github.com/LukeZhuang/Hierarchical-Attention-Network/blob/master/yelp-preprocess.ipynb
    :param sentence:  is a str
    :return:
    """
    nonalpnum = re.compile('[^0-9a-zA-Z?!\']+')
    words = sentence.split()
    words_collection=[]
    for word in words:
        if word in ['-lrb-','-rrb-','<sssss>']:
            continue
        tt=nonalpnum.split(word)
        t=''.join(tt)
        if t!='':
            words_collection.append(t)

    return words_collection


def clean_stopwords(word_list):   # seems unreasonable
    """
    remove stopwords, not sure if it's useful
    :param word_list: list of word
    :return:
    """
    stopwords=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
               'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
               'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
               "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
               'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
               'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
               'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
               'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
               'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
               'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
               'through', 'during', 'before', 'after', 'above', 'below', 'to',
               'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
               'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
               'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
               'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
               'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
               "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
               'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
               'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
               "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
               'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
               "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    # or import from nltk
    # stopwords.words('english'),
    # but some problems occur in my nltk version
    import string
    stoplist=stopwords+list(string.punctuation)

    return [word for word in word_list if word not in stoplist]

def stem(word_list):
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer('english')
    return [stemmer.stem(word) for word in word_list]

def all_text(docs):
    texts=[]
    for doc in docs:
        for sentences in doc:
            for sent in sentences:
                texts.append(sent)
    return texts


