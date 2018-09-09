import numpy as np


def load_data(data_path):
    data = []
    sentence = []
    labels = []
    word_count = 0
    for line in open(data_path, 'rb'):
        if line != '':
            word, label = line.strip().split()
            sentence.append(word)
            labels.append(label)
            word_count += 1
        else:
            if len(sentence):
                data.append((sentence, labels))
                sentence.clear()
                labels.clear()
    if len(sentence):
        data.append((sentence, labels))
    print('num of word : {}'.format(word_count))
    print('num of sentence : {}'.format(len(data)))
    return data


def build_label_vocab(data_path):
    label_list = []
    for line in open(data_path, 'rb'):
        line = line.strip()
        label_list.append('B-' + line)
        label_list.append('I-' + line)
    label2idx = dict(zip(label_list, range(label_list)))
    idx2label = dict(zip(range(label_list), label_list))
    return label2idx, idx2label


def build_word_vocab(data, min_count=0):
    word_count = [('UNK', -1), ('PAD', -1)]
    word_list = []
    for sentence, labels in data:
        word_list.extend(sentence)
    from collections import Counter, defaultdict
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


def build_dataset(data, word2idx, label2idx):
    num_text = []
    num_label = []
    for sentence, label in data:
        num_text.append(np.array([word2idx[w] for w in sentence], dtype=np.int64))
        num_label.append(np.array([word2idx[w] for w in sentence], dtype=np.int64))
    return num_text, num_label
