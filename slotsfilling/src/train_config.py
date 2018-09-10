import os

from commons import TAG2LABEL, WORD2ID, LABEL2TAG, ID2WORD, get_logger
from project_config import LOG_DIR
import numpy as np


class TrainConfig(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.clip = args.clip
        self.demo_model = args.demo_model
        self.dropout = args.dropout
        self.embedding_dim = args.embedding_dim
        self.epoch = args.epoch
        self.epochs_no_impv = args.epochs_no_impv
        self.hidden_size_lstm = args.hidden_size_lstm
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.mode = args.mode
        self.model_dir = args.model_dir
        self.nwords = len(WORD2ID)
        self.nlabels = len(TAG2LABEL)
        self.optimizer = args.optimizer
        if args.pretrain_embeddings == 'random':
            self.pretrain_embeddings = None
        else:
            data = np.load(args.pretrain_embeddings)
            self.pretrain_embeddings = data['embeddings']

        self.shuffle = args.shuffle
        self.summary_dir = args.summary_dir
        self.test_data = args.test_data
        self.train_data = args.train_data
        self.train_embedding = args.train_embedding
        self.use_crf = args.use_crf

        self.word2id = WORD2ID
        self.tag2label = TAG2LABEL
        self.id2word = ID2WORD
        self.label2tag = LABEL2TAG

        self.logger = get_logger(os.path.join(LOG_DIR, 'train.log'))
