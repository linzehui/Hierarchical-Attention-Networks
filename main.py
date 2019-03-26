import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from model.HierAttnModel import *
import torch.optim as optim
from train import *
from config import *


torch.manual_seed(1)

def main():
    # --- prepare dataset --- #
    train_file=ops.train_file
    valid_file=ops.valid_file
    test_file=ops.test_file

    train_docs_dataset,valid_docs_dataset,test_docs_dataset,train_docs,valid_docs,test_docs,vocab=\
        get_dataset(train_file,valid_file,test_file,ops)  # docs are raw data

    train_loader,valid_loader,test_loader=get_dataloader(train_docs_dataset,valid_docs_dataset,test_docs_dataset,ops)


    # --- prepare model,etc --- #
    device=ops.device

    pretrained_matrix=None
    if ops.use_pretrain:
        # construct from model
        model_file='word2vec'+train_file
        import os
        if not os.path.exists(model_file):   # then we have to construct it
            print('[Info] Training word2vec model from scratch...')
            texts=all_text([train_docs,valid_docs,test_docs])
            word2vec_model=train_word2vec(texts,model_file)
        else:
            import gensim
            print('[Info] Loading word2vec model')
            word2vec_model=gensim.models.word2vec.Word2Vec.load(model_file)

        pretrained_matrix=construct_embedding_matrix_from_model(word2vec_model,vocab)

        # pretrained_matrix=construct_embedding_matrix_from_file('/home/zhlin/HAN/glove.6B.200d.txt',
        #                                                        vocab=vocab)

    model=HierAttnModel(
        vocab_size=len(vocab),
        word_hidden_dim=ops.word_hidden_dim,
        sent_hidden_dim=ops.sent_hidden_dim,
        embed_dim=ops.embedding_dim,
        n_label=ops.n_label,
        pretrained_matrix=pretrained_matrix,
        attn_dropout=ops.attn_dropout
    ).to(device)


    criterion=torch.nn.CrossEntropyLoss()
    if ops.optim == 'SGD':
        optimizer=optim.SGD(model.parameters(),lr=ops.lr,momentum=ops.momentum)
    elif ops.optim == 'Adam':
        optimizer=optim.Adam(model.parameters(),lr=ops.lr)

    print('[Model]',model)
    summary_writer = SummaryWriter('experiment' + str(ops.experiment_id))

    # -------- #
    train(model,train_loader,valid_loader,test_loader,
          optimizer,criterion,device,ops,summary_writer)


if __name__=='__main__':
    main()

