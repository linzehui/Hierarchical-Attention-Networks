import torch


class Config():
    # ----cuda setting----#
    use_cuda = torch.cuda.is_available()
    CUDA_VISIBLE_DEVICES = [1]
    device = torch.device('cuda:' + str(
        CUDA_VISIBLE_DEVICES[0]) if use_cuda else 'cpu')

    # --- data settting --- #
    min_freq = 5
    embedding_dim = 200
    use_pretrain = True
    freeze = True
    max_sent_len = 100
    max_num_sent = 100
    min_num_sent = 1
    n_label = 5

    train_file = 'yelp-2014-train.txt.ss'
    # train_file = 'yelp-2014-train.txt.ss'
    valid_file = 'yelp-2014-dev.txt.ss'
    test_file = 'yelp-2014-test.txt.ss'

    sampler = 'NumSentence'  # 'Random' 'NumSentence' 'MaxSentence'
    reverse = True


    # --- RNN setting --- #
    word_hidden_dim = 50
    sent_hidden_dim = 50
    attn_dropout = 0

    # --- training setting--- #
    epoch = 100
    batch_size = 256
    optim = 'SGD'  # 'Adam' or 'SGD'
    lr = 2e-4
    momentum = 0.9
    clip = 0.5
    interval = 3

    experiment_id = 104


ops = Config
