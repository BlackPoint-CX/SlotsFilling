import tensorflow as tf
import os


class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.logger = config.logger

        self.sess = None
        self.saver = None

    def init_session(self):
        self.logger.info('Initialize tf session.')
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def save_session(self):
        self.logger.info('Save tf session.')
        if not os.path.exists(self.config.model_dir):
            os.makedirs(self.config.model_dir)
        self.saver.save(self.sess, self.config.model_dir)

    def cloes_sess(self):
        self.logger.info('Close tf session.')
        self.sess.close()

    def restore_sess(self, model_path):
        self.logger.info('Restore latest model.')
        self.saver.restore(self.sess, model_path)

    def add_summary(self):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.summary_dir, self.sess.graph)

    def reinitialize_weigths(self, scope_name):
        """
        Reinitialize weights of specific layer.
        :param scope_name: name of variable scope.
        """
        self.logger.info('Reinitialize weights in {}.'.format(scope_name))
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)

    def add_train_op(self, optimizer, lr, loss, clip=-1):
        _optimizer = optimizer.lower()

        with tf.variable_scope('train_step'):
            if _optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(lr)
            elif _optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _optimizer == 'adagrad':
                optimizer = tf.train.AdagradDAOptimizer(lr)
            elif _optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError('Optimizer cannot be recognized. Plz check again.')

            if clip > 0:
                gradients, variables = zip(*optimizer.compute_gradients(loss))
                gradients, gradient_norms = tf.clip_by_global_norm(gradients, clip)
                self.train_op = optimizer.apply_gradients(zip(gradients, variables))
            else:
                self.train_op = optimizer.minimize(loss)

    def train(self, train, dev):
        best_score = 0
        epochs_no_impv = 0
        self.add_summary()

        for epoch_idx in range(self.config.epoch):
            self.logger.info('Running Epoch {}.'.format(epoch_idx))
            score = self.run_epoch(train, dev, epoch_idx)
            self.config.lr *= self.config.lr_decay
            if score > best_score:
                epochs_no_impv = 0
                self.save_session()
                best_score = score
                self.logger.info('New Best Score : {}'.format(best_score))
            else:
                epochs_no_impv += 1
                if epochs_no_impv > self.config.epochs_no_impv:
                    self.logger.info('No Improvement. Break. ')
                    break

    def evaluate(self, test):
        self.logger.info('Testing model on test dataset.')
        metrics = self.run_evaluate(test)
        msg = '-'.join(['{} {:04.2f}'.format(k, v) for k, v in metrics.items()])
        self.logger.info(msg)
