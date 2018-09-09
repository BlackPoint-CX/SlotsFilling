from copy import deepcopy

from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode

from base_model import BaseModel
import tensorflow as tf
import numpy as np


class Progbar(object):
    pass


class SlotFillingModel(BaseModel):
    def __init__(self, config):
        super(SlotFillingModel, self).__init__()
        self.idx2tag = deepcopy(self.config.idx2tag)

    def build(self):
        self.add_placeholder_op()
        self.add_word_embedding_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

        self.add_train_op(lr_method=self.config.lr_method, lr=self.config.lr, loss=self.loss, clip=self.config.clip)

        self.init_session()

    def add_placeholder_op(self):
        # Shape of (batch_size, max length of sequences)
        self.word_ids = tf.placeholder(dtype=tf.float32, shape=[None, None], name='word_ids')

        # Shape of (batch_size)
        self.sequence_lengths = tf.placeholder(dtype=tf.float32, shape=[None], name='sequence_length')

        # Shape of (batch_size, max length of sequences, max length of word)
        self.char_ids = tf.placeholder(dtype=tf.float32, shape=[None, None, None], name='char_ids')

        # Shape of (batch_size, max length of sentences)
        self.word_lengths = tf.placeholder(dtype=tf.float32, shape=[None, None], name='word_length')

        # Shape of (batch_size, max length of sentence in batch)
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, None], name='labels')

        # Hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name='dropout')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='learning rate')

    def add_word_embedding_op(self):
        with tf.variable_scope('add_word_embeddings_op_words'):
            if self.config.embeddings is None:
                self.logger.info('Warning : Random Initialze Embeddings.')
                _word_embeddings = tf.get_variable(name='_word_embeddings',
                                                   shape=[self.config.nwords, self.config.dim_word], dtype=tf.float32)
            else:
                _word_embeddings = tf.Variable(initial_value=self.config.embeddings, name='_word_embeddings',
                                               dtype=tf.float32, trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(parmas=_word_embeddings, ids=self.word_ids, name='word_embeddings')

        with tf.variable_scope('add_word_embeddings_op_chars'):
            if self.config.use_chars:
                _char_embeddings = tf.get_variable(name='_char_embeddings',
                                                   shape=[self.config.nchars, self.config.nchars], dtype=tf.float32, )
                char_embeddings = tf.nn.embedding_lookup(params=_char_embeddings, ids=tf.char_idx,
                                                         name='char_embeddings')

                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(tensor=char_embeddings, shape=[s[0] * s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(tensor=self.word_lengths, shape=[s[0] * s[1]])

                cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size_char,
                                                  state_is_tuple=True)
                cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=self.config.hidden_size_char, state_is_tuple=True)

                _output = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=char_embeddings,
                                                          sequence_length=word_lengths, dtype=tf.float32)

                ((output_fw, output_bw), (output_state_fw, output_state_bw)) = _output

                # _, ((_, output_fw), (_, output_bw)) = _output

                output = tf.concat(values=[output_fw, output_bw], axis=-1)

                output = tf.reshape(tensor=output, shape=[s[0], s[1], 2 * self.config.hidden_size_char])

                word_embeddings = tf.concat(values=[word_embeddings, output], axis=-1)

        self.word_embeddings = tf.nn.dropout(x=word_embeddings, keep_prob=self.dropout)

    def add_logits_op(self):
        with tf.variable_scope('add_logits_op_bi_lstm'):
            cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=tf.config.hidden_size_lstm)
            cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=tf.config.hidden_size_lstm)
            ((output_fw, output_bw), (output_state_fw, output_state_bw)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=1)
            output = tf.nn.dropout(x=output, keep_prob=tf.config.dropout)

        with tf.variable_scope('proj'):
            W = tf.get_variable(name='W', shape=[2 * self.config.hidden_size_lstm, self.config.ntags])
            b = tf.get_variable(name='b', shape=[self.config.ntags])
            nsteps = tf.shape(output)[1]
            output = tf.reshape(tensor=output, shape=[-1, 2 * self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b

        self.logits = tf.reshape(tensor=pred, shape=[-1, nsteps, self.config.ntags])

    def add_pred_op(self):
        if not self.config.use_crf:
            self.labels_pred = tf.cast(x=tf.argmax(self.logits, axis=-1), dtype=tf.float32)

    def add_loss_op(self):
        if self.config.use_crf:
            log_likelihood, trans_params = crf_log_likelihood(inputs=self.word_ids, tag_indices=self.labels,
                                                              sequence_lengths=self.sequence_lengths)
            self.trans_params = trans_params
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(tensor=losses, mask=mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar('loss', self.loss)

    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, pad_token=0, nlevel=1)
            char_ids, word_lengths = pad_sequences(char_ids, pad_token=0, nlevel=2)
        else:
            word_ids, sequences = pad_sequences(words, 0)

        feed_dict = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_chars:
            feed_dict[self.char_ids] = char_ids
            feed_dict[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, pad_token=0)
            feed_dict[self.labels] = labels

        if lr is not None:
            feed_dict[self.lr] = lr

        if dropout is not None:
            feed_dict[self.dropout] = dropout

        return feed_dict, sequence_lengths

    def predict_batch(self, words):
        fd, sequence_lengths = self.get_feed_dict(words=words, dropout=1.0)
        if self.config.use_crf:
            viterbi_sequences = []
            logits, trans_params = self.sess.run(fetchs=[self.logits, self.trans_params], feed_dict=fd)

            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length]
                viterbi_seq, viterbi_score = viterbi_decode(score=logit, transition_params=trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths
        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)
            return labels_pred, sequence_lengths

    def run_epoch(self, train, dev, epoch):
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, sequence_lengths = self.get_feed_dict(words=words, labels=labels, lr=self.config.lr,
                                                      dropout=self.config.dropout)
            _, train_loss, summary = self.sess.run(fetchs=[self.train_op, self.loss, self.merged], feed_dict=fd)
            prog.update(i + 1, [('train_loss', train_loss)])

            if i % 10 == 0:
                self.file_writer.add_summary(summary=summary, global_step=epoch * batch_size + 1)

        metrics = self.run_evaluate(dev)
        msg = '-'.join(['{}:{:04.2f}'.format(k, v) for k, v in metrics.items()])
        self.logger.info(msg)
        return metrics['f1']

    def run_evaluate(self, test):
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):

            label_preds, sequence_lengths = self.predict_batch(words=words)

            for label, label_pred, length in zip(labels, label_preds, sequence_lengths):
                label = label[:length]
                label_pred = label_pred[:length]

                accs += [label == pred for (label, pred) in zip(label, label_pred)]
                label_chunks = set(get_chunks(label, self.config.tag_idx))
                label_pred_chunks = set(get_chunks(label_pred, self.config.tag_idx))

                correct_preds += len(label_chunks & label_pred_chunks)
                total_preds += len(label_pred_chunks)
                total_correct += len(label_chunks)

        p = correct_preds / total_preds
        r = correct_preds / total_correct
        f1 = 2 * p * r / (p + r)
        accs = np.mean(accs)
        return {'accs': accs, 'f1': f1}


def get_chunks(seq, tag_idx):
    """

    :param seq:
    :param tag_idx:
    :return:

    Example :
        seq : [1,2,0,3]
        tag_idx : {'B-PER' : 1, 'I-PER': 2, 'B-ORG' : 3, 'O' : 0}
        chunks : [('PER', 0 ,2), ('ORG',3,4)]
    """
    NONE = 'O'
    default = tag_idx[NONE]
    idx2tag = {idx: tag for tag, idx in tag_idx.items()}
    chunks = []
    chunk_type, chunk_start = None, None  # These two should be changed together.
    for i, tok in enumerate(seq):
        if tok == default:  # Cut by 'O'
            if chunk_type is None:
                pass
            else:
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None
        else:
            tok_BIO, tok_type = get_chunk_type(tok, idx2tag)
            if chunk_type is None:
                if tok_BIO == 'B':
                    chunk_type = tok_type
                    chunk_start = i
                else:
                    pass  # wrong case of  'O O I-PER I-PER'
            else:
                if tok_BIO == 'B':
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                    chunk_type = tok_type
                    chunk_start = i
                elif tok_BIO == 'O':
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                    chunk_type, chunk_start = None, None
                else:
                    # tok_BIO = 'I'
                    if tok_type == chunk_type:
                        pass
                    else:
                        chunk = (chunk_type, chunk_start, i)
                        chunks.append(chunk)
                        chunk_type, chunk_start = None, None

    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


def get_chunk_type(idx, idx2tag):
    tag = idx2tag(idx)
    tag_BIO, tag_type = tag.split('-')
    return tag_BIO, tag_type


def minibatches(data, minibatch_size):
    pass


def pad_sequences(sequences, pad_token, nlevel=1):
    pass
