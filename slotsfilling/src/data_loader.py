from random import random
from commons import TAG2LABEL, LABEL2TAG, WORD2ID, ID2WORD


class DataManager(object):
    def __init__(self, data_file_path, processing_word=None, prosessing_tag=None):
        self.data_file_path = data_file_path
        self.processing_word = processing_word
        self.processing_tag = prosessing_tag
        self.length = None
        self.origin_data = self.build_data()
        self.preprocess_data = self.preprocess_data()

    def build_data(self):
        """
        Build word and tag.
        :return:
        """
        data = []
        sentence_words, sentence_tags = [], []
        with open(self.data_file_path) as r_file:
            for line in r_file:
                if line == '\n':
                    data.append((sentence_words, sentence_tags))
                    sentence_words, sentence_tags = [], []
                else:
                    word, tag = line.strip().split()
                    sentence_words.append(word)
                    sentence_tags.append(tag)
            if len(sentence_words):
                data.append((sentence_words, sentence_tags))
        return data

    def preprocess_data(self):
        """
        Transfer word+tag to id+label.
        :return:
        """
        data = []
        for (sentence_words, sentence_tags) in self.origin_data:
            sentence_ids = []
            sentence_labels = []
            for word in sentence_words:
                word_id = WORD2ID[word]
                sentence_ids.append(word_id)
            for tag in sentence_tags:
                label = TAG2LABEL[tag]
                sentence_labels.append(label)
            data.append((sentence_ids, sentence_labels))
        return data

    def __iter__(self, batch_size, shuffle=False):
        if shuffle:
            data = random.shuffle(self.preprocess_data)
        else:
            data = self.preprocess_data

        ids, labels = [], []
        for (sentence_ids, sentence_labels) in data:
            ids.append(sentence_ids)
            labels.append(sentence_labels)
            if len(ids) == batch_size:
                yield ids, labels
                ids, labels = [], []

        if len(ids):
            yield ids, labels

    def __len__(self):
        if self.length is None:
            self.length = 0
            for _ in self.origin_data:
                self.length += 1
        return self.length
