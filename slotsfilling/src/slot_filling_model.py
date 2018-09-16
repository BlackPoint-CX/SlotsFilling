from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode

from base_model import BaseModel
import tensorflow as tf
import numpy as np
from progressbar import ProgressBar

from commons import get_chunks, pad_sequences, WORD2ID, LABEL2TAG, TAG2LABEL


class SlotFillingModel(BaseModel):
    def __init__(self, config):
        super(SlotFillingModel, self).__init__(config)
        self.config = config

    def build(self):
        self.add_placeholder_op()
        self.add_word_embedding_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op(optimizer=self.config.optimizer, lr=self.config.lr, loss=self.loss, clip=self.config.clip)
        self.init_session()

    def add_placeholder_op(self):
        # Shape of (batch_size, max length of sequences)
        self.word_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='word_ids')

        # Shape of (batch_size)
        self.sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence_length')

        # Shape of (batch_size, max length of sentence in batch)
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, None], name='labels')

        # Hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name='dropout')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')

    def add_word_embedding_op(self):
        with tf.variable_scope('add_word_embeddings_op_words'):
            if self.config.pretrain_embeddings is None:
                self.logger.info('Warning : Random Initialize Embeddings.')
                _word_embeddings = tf.get_variable(name='_word_embeddings',
                                                   shape=[self.config.nwords, self.config.embedding_dim],
                                                   dtype=tf.float32)
            else:
                self.logger.info('Using Pretrain Embeddings.')
                _word_embeddings = tf.Variable(initial_value=self.config.pretrain_embeddings,
                                               name='_word_embeddings',
                                               dtype=tf.float32, trainable=self.config.train_embedding,
                                               expected_shape=[self.config.nwords, self.config.embedding_dim])
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=self.word_ids, name='word_embeddings')
        self.word_embeddings = tf.nn.dropout(x=word_embeddings, keep_prob=self.dropout)

    def add_logits_op(self):
        with tf.variable_scope('add_logits_op_bi_lstm'):
            cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=self.config.hidden_size_lstm)
            cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=self.config.hidden_size_lstm)
            ((output_fw, output_bw), (output_state_fw, output_state_bw)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, cell_bw=cell_bw, inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(x=output, keep_prob=self.config.dropout)
            print('output.shape after bi_lstm', output.shape)

        with tf.variable_scope('proj'):
            W = tf.get_variable(name='W', shape=[2 * self.config.hidden_size_lstm, self.config.nlabels])
            b = tf.get_variable(name='b', shape=[self.config.nlabels])
            nsteps = tf.shape(output)[1]

            output = tf.reshape(tensor=output, shape=[-1, 2 * self.config.hidden_size_lstm])
            print('output.shape', output.shape)
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(tensor=pred, shape=[-1, nsteps, self.config.nlabels])
            print('logits.shape', self.logits.shape)

    def add_pred_op(self):
        if not self.config.use_crf:
            self.labels_pred = tf.cast(x=tf.argmax(self.logits, axis=-1), dtype=tf.float32)

    def add_loss_op(self):
        if self.config.use_crf:
            log_likelihood, trans_params = crf_log_likelihood(inputs=self.logits, tag_indices=self.labels,
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

        word_ids, sequence_lengths = pad_sequences(words, 0)

        feed_dict = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if labels is not None:
            labels, _ = pad_sequences(labels, pad_token=0)
            feed_dict[self.labels] = labels

        if lr is not None:
            feed_dict[self.lr] = lr

        if dropout is not None:
            feed_dict[self.dropout] = dropout

        return feed_dict, sequence_lengths

    def predict_batch(self, words):
        fd, sequence_lengths = self.get_feed_dict(words=words, dropout=0.5)
        if self.config.use_crf:
            viterbi_sequences = []
            logits, trans_params = self.sess.run(fetches=[self.logits, self.trans_params], feed_dict=fd)
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length]
                viterbi_seq, viterbi_score = viterbi_decode(score=logit, transition_params=trans_params)
                viterbi_sequences += [viterbi_seq]
            labels_pred = viterbi_sequences
        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

        return labels_pred, sequence_lengths

    def run_epoch(self, train, dev, epoch):
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = ProgressBar(nbatches).start()

        for i, (words, labels) in enumerate(train):
            fd, sequence_lengths = self.get_feed_dict(words=words, labels=labels, lr=self.config.lr,
                                                      dropout=self.config.dropout)
            _, train_loss, summary = self.sess.run(fetches=[self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1)
            self.logger.info('train_loss : {} at batch {}'.format(train_loss, i))

            if i % 10 == 0:
                self.file_writer.add_summary(summary=summary, global_step=epoch * batch_size + 1)

        metrics = self.run_evaluate(dev)
        msg = '-'.join(['{}:{:04.2f}'.format(k, v) for k, v in metrics.items()])
        self.logger.info(msg)
        return metrics['f1']

    def run_evaluate(self, test):
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in test:

            label_preds, sequence_lengths = self.predict_batch(words=words)

            for label, label_pred, length in zip(labels, label_preds, sequence_lengths):
                label = label[:length]
                label_pred = label_pred[:length]

                accs += [label == pred for (label, pred) in zip(label, label_pred)]
                label_chunks = set(get_chunks(label, TAG2LABEL))
                label_pred_chunks = set(get_chunks(label_pred, TAG2LABEL))

                correct_preds += len(label_chunks & label_pred_chunks)
                total_preds += len(label_pred_chunks)
                total_correct += len(label_chunks)

        p = correct_preds / total_preds
        r = correct_preds / total_correct
        f1 = 2 * p * r / (p + r)
        accs = np.mean(accs)
        return {'accs': accs, 'f1': f1}

    def predict(self, words_raw):
        words = [WORD2ID(w) for w in words_raw]
        pred_ids, sequence_lengths = self.predict_batch([words])
        # Translate id with associate tag
        preds = [LABEL2TAG[pred_id] for pred_id in pred_ids]
        return preds
