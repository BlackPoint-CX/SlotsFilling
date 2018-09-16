from argparse import ArgumentParser
import os

from commons import str2bool, TAG2LABEL, get_chunks
from data_manager import DataManager
from project_config import DATA_DIR, SUMMARY_DIR, MODEL_DIR
from slot_filling_model import SlotFillingModel
from train_config import TrainConfig
import tensorflow as tf


def main(config):
    train_data = DataManager(os.path.join(DATA_DIR, config.train_data))
    test_data = DataManager(os.path.join(DATA_DIR, config.test_data))

    if config.mode == 'train':
        model = SlotFillingModel(config=config)
        model.build()

        model.train(train=train_data, dev=test_data)
        model.save_session()


    elif config.mode == 'test':
        ckpt_file = tf.train.latest_checkpoint(config.model_dir)
        print('ckpt_file:', ckpt_file)
        model = SlotFillingModel(config=config)
        model.build()
        model.restore_sess(ckpt_file)
        evaluates = model.run_evaluate(test_data)
        print(repr(evaluates))

    elif config.mode == 'demo':
        ckpt_file = tf.train.latest_checkpoint(config.model_dir)
        print('ckpt_file:',ckpt_file)
        model = SlotFillingModel(config=config)
        model.build()
        model.restore_sess(ckpt_file)
        print('============= demo =============')
        # i would like to find a flight from charlotte to las vegas that makes a stop in st. louis
        while True:
            print('Please input your sentence:')
            demo_sent = input()
            if demo_sent == '' or demo_sent.isspace():
                print('See you next time!')
                break
            else:
                demo_sent = list(demo_sent.strip().split())
                pred_ids, tags = model.predict(demo_sent)
                chunks = get_chunks(pred_ids,  TAG2LABEL)
                print(chunks)

    else:
        raise ValueError('Wrong Mode')


if __name__ == '__main__':
    parser = ArgumentParser(description='BiLSTM-CRF for Slot Filling')

    parser.add_argument('--batch_size', type=int, default=20, help='num of records in each batch')
    parser.add_argument('--clip', type=float, default=1.0, help='gradient clipping')
    parser.add_argument('--demo_model', type=str, default='', help='model for test and demo')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
    parser.add_argument('--embedding_dim', type=int, default=300, help='dim of embedding')
    parser.add_argument('--epoch', type=int, default=50, help='num of epochs')
    parser.add_argument('--epochs_no_impv', type=int, default=8, help='epochs without improvement')
    parser.add_argument('--hidden_size_lstm', type=int, default=128, help='dim of hidden layer')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.05, help='learning rate decay')
    parser.add_argument('--mode', type=str, default='demo', help='[train test demo]')
    parser.add_argument('--model_dir', type=str, default=MODEL_DIR + '/', help='dir of model')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer(low case better)[adam sgd adagrad rmsprop]')
    parser.add_argument('--pretrain_embeddings', type=str, default='random',
                        help='use pretrained embedding or init it randomly, input path of embedding file or just \'random\'')
    parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
    parser.add_argument('--summary_dir', type=str, default=SUMMARY_DIR, help='dir of summary')
    parser.add_argument('--test_data', type=str, default='atis.test.txt', help='test data path')
    parser.add_argument('--train_data', type=str, default='atis.train.txt', help='train data path')
    parser.add_argument('--train_embedding', type=str2bool, default=False, help='update embedding during training')
    parser.add_argument('--use_crf', type=str2bool, default=True, help='using crf')

    train_config = TrainConfig(args=parser.parse_args())
    main(config=train_config)
