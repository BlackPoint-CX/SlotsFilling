from argparse import ArgumentParser
import os

from data_loader import DataManager
from project_config import DATA_DIR


def main(parser):
    train_data = DataManager(os.path.join(DATA_DIR, 'atis.train.txt'))
    test_data = DataManager(os.path.join(DATA_DIR, 'atis.test.txt'))


if __name__ == '__main__':
    parser = ArgumentParser(description='BiLSTM-CRF for Slot Filling')
    parser.add_argument('--train_data', type=str, default='atis.train.txt', help='train data path')
    parser.add_argument('--test_data', type=str, default='atis.test.txt', help='test data path')
    parser.add_argument('--batch_size', type=int, default=20, help='num of records in each batch')
    parser.add_argument('--epoch', type=int, default=50, help='num of epochs')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer(low case better)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--use_crf', type=str2bool, default=True, help='using crf')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
    parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
    parser.add_argument('--hidden_dim', type=int, default=128, help='dim of hidden layer')
    parser.add_argument('--clip', type=float, default=1.0, help='gradient clipping')
    parser.add_argument('--train_embedding', type=str2bool, default=False, help='update embedding during training')
    parser.add_argument('--pretrain_embedding', type=str, default='random',
                        help='use pretrained embedding or init it randomly(ramdom)')
    parser.add_argument('--embedding_dim', type=int, default=300, help='dim of embedding')
    parser.add_argument('--mode', type=str, default='demo', help='[train test demo]')
    parser.add_argument('--demo_model', type=str, default='', help='model for test and demo')

    main(parser=parser)
