import argparse
import logging
import os
import numpy as np
from project_config import DATA_DIR
from collections import Counter, defaultdict


def build_label_vocab(data_path):
    """
    Read label file and assign label to each BIO-tag.
    :param data_path:
    :return:
    """
    tag_list = []
    for line in open(data_path, 'r'):
        line = line.strip()
        tag_list.append('B-' + line)
        tag_list.append('I-' + line)
    tag_list.append('O')
    tag2label = dict(zip(tag_list, range(len(tag_list))))
    label2tag = dict(zip(range(len(tag_list)), tag_list))
    return tag2label, label2tag


TAG2LABEL, LABEL2TAG = build_label_vocab(os.path.join(DATA_DIR, 'atis_slot_names.txt'))


def build_word_vocab(data_file_path, min_count=0):
    """
    Read all file and assign id to each word
    :param data_file_path:
    :param min_count:
    :return:
    """
    word_count = [('UNK', -1), ('PAD', -1)]
    word_list = []
    for line in open(data_file_path, 'r'):
        if line not in ['', '\n', '']:
            word, tag = line.strip().split('\t')
            word_list.append(word)

    counter = Counter(word_list)
    counter_freq = counter.most_common()
    for word, freq in counter_freq:
        if freq > min_count:
            word_count.append((word, freq))
    word2id = defaultdict(int)
    for word, count in word_count:
        word2id[word] = len(word2id)

    id2word = dict(zip(word2id.values(), word2id.keys()))
    return word2id, id2word


WORD2ID, ID2WORD = build_word_vocab(os.path.join(DATA_DIR, 'atis.all.txt'))


def get_chunks(seq, tag2label):
    """
    Identify each type of entity/slot.
    :param seq:
    :param tag2label:
    :return:

    Example :
        seq : [1,2,0,3]
        tag_idx : {'B-PER' : 1, 'I-PER': 2, 'B-ORG' : 3, 'O' : 0}
        chunks : [('PER', 0 ,2), ('ORG',3,4)]
    """
    NONE = 'O'
    default = tag2label[NONE]
    label2tag = {label: tag for tag, label in tag2label.items()}
    chunks = []
    chunk_type, chunk_start = None, None  # These two should be changed together.
    for i, label in enumerate(seq):
        if label == default:  # Cut by 'O'
            if chunk_type is None:
                pass
            else:
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None
        else:
            tag_BIO, tag_type = get_chunk_type(label, label2tag)
            if chunk_type is None:
                if tag_BIO == 'B':
                    chunk_type = tag_type
                    chunk_start = i
                else:
                    pass  # wrong case of  'O O I-PER I-PER'
            else:
                if tag_BIO == 'B':
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                    chunk_type = tag_type
                    chunk_start = i
                elif tag_BIO == 'O':
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                    chunk_type, chunk_start = None, None
                else:
                    # tag_BIO = 'I'
                    if tag_type == chunk_type:
                        pass
                    else:
                        chunk = (chunk_type, chunk_start, i)
                        chunks.append(chunk)
                        chunk_type, chunk_start = None, None

    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def get_chunk_type(label, label2tag):
    """
    Check BIO and type of tag.
    :param label:
    :param label2tag:
    :return:
    """
    tag = label2tag[label]
    tag_BIO, tag_type = tag.split('-')
    return tag_BIO, tag_type


def pad_sequences(sequences, pad_token):
    max_length = max(map(lambda x: len(x), sequences))
    padded_seq, sequence_lengths = [], []
    for seq in sequences:
        seq = list(seq)
        sequence_lengths.append(len(seq))
        seq.extend([pad_token] * (max_length - len(seq)))
        padded_seq.append(seq)
    return padded_seq, sequence_lengths


def str2bool(word):
    word = word.lower()
    if word in ['yes', 'true', 'y', '1']:
        return True
    elif word in ['no', 'false', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename=filename)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    handler.setFormatter(fmt=formatter)
    logging.getLogger().addHandler(hdlr=handler)
    return logger


if __name__ == '__main__':
    print(WORD2ID)
    print(TAG2LABEL)
