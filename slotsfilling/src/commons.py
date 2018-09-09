import argparse
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
    tag2label = dict(zip(tag_list, range(len(tag_list))))
    label2tag = dict(zip(range(len(tag_list)), tag_list))
    return tag2label, label2tag


TAG2LABEL, LABEL2TAG = build_label_vocab(os.path.join(DATA_DIR, 'atis_slot_names.txt'))


def build_word_vocab(data, min_count=0):
    """
    Read all file and assign id to each word
    :param data:
    :param min_count:
    :return:
    """
    word_count = [('UNK', -1), ('PAD', -1)]
    word_list = []
    for word, labels in data:
        word_list.extend(word)
    counter = Counter(word_list)
    counter_freq = counter.most_common()
    for word, freq in counter_freq:
        if freq > min_count:
            word_count.append((word, freq))
    word2idx = defaultdict(int)
    for word, count in word_count:
        word2idx[word] = len(word2idx)

    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    return word2idx, idx2word


WORD2ID, ID2WORD = build_word_vocab(os.path.join(DATA_DIR, 'atis.all.txt'))


def build_dataset(data, word2idx, label2idx):
    num_text = []
    num_label = []
    for sentence, label in data:
        num_text.append(np.array([word2idx[w] for w in sentence], dtype=np.int64))
        num_label.append(np.array([word2idx[w] for w in sentence], dtype=np.int64))
    return num_text, num_label


def get_chunks(seq, tag_label):
    """
    Identify each type of entity/slot.
    :param seq:
    :param tag_label:
    :return:

    Example :
        seq : [1,2,0,3]
        tag_idx : {'B-PER' : 1, 'I-PER': 2, 'B-ORG' : 3, 'O' : 0}
        chunks : [('PER', 0 ,2), ('ORG',3,4)]
    """
    NONE = 'O'
    default = tag_label[NONE]
    label2tag = {idx: tag for tag, idx in tag_label.items()}
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
    tag = label2tag(label)
    tag_BIO, tag_type = tag.split('-')
    return tag_BIO, tag_type


def pad_sequences(sequences, pad_token, nlevel=1):
    pass


def str2bool(word):
    word = word.lower()
    if word in ['yes','true','y','1']:
        return True
    elif word in ['no','false','n','0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
