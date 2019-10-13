import tensorflow as tf
from tensorflow.contrib import rnn
import sys

sys.path.append('..')
sys.path.append('.')

from utils import constants
from utils import optim


class Regressor(object):
    def __init__(self, mode, params, vocab_size=None):
        self.params = params
        self.mode = mode
        self.hidden_size = params.get("hidden_size", 128)
        self.num_layers = params.get("num_layers")
        self.emb_dim = params.get("emb_dim")
        self.keep_prob = params.get("keep_prob") if mode == constants.TRAIN else 1.0
        self.learning_rate = params.get("learning_rate", 0.001)
        self.vocab_size = vocab_size
        self.bidirectional = params.get("bidirectional", True)
        self.grad_clip = params.get("clip_gradients")
        self.has_attention = params.get("has_attention", False)
        self.scale_sentiment = params.get("scale_sentiment", True)
        self.sigmoid_pred_score = params.get("sigmoid_pred_score", False)

        self.x = tf.placeholder(tf.int32, shape=(None, None), name=constants.INPUT_IDS)
        self.y = tf.placeholder(tf.float32, shape=(None,), name=constants.LABEL_OUT)
        # the original sequence length (except for bos and eos tag)
        self.sequence_length = tf.placeholder(tf.int32, shape=(None,), name=constants.LENGTH)

        self.batch_size = tf.shape(self.x)[0]
        self.max_len = tf.shape(self.x)[1]

        # self.initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode="FAN_AVG", uniform=True)
        self.initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("word_embedding", initializer=self.initializer):
            with tf.device("/cpu:0"):
                self.embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.emb_dim], trainable=True)

        if params["rnn_cell"].lower() == "rnn":
            self.cell_fn = rnn.BasicRNNCell
        elif params["rnn_cell"].lower() == "gru":
            self.cell_fn = rnn.GRUCell
        elif params["rnn_cell"].lower() == "lstm":
            self.cell_fn = rnn.LSTMCell
        else:
            raise Exception("model type not supported: {}".format(params.model))

        if self.bidirectional:
            output = self.build_birnn()
        else:
            output = self.build_unirnn()
        output = tf.nn.dropout(output, self.keep_prob)  # [batch_size, output_size]

        with tf.variable_scope('output_projection', initializer=self.initializer):
            self.output_w = tf.get_variable("output_w", [self.hidden_size, 1])
            self.output_b = tf.get_variable("output_b", [1])
            self.logits = tf.nn.xw_plus_b(output, self.output_w, self.output_b)
            if self.sigmoid_pred_score:
                print("sigmoid_pred_score")
                self.predict_score = tf.sigmoid(tf.squeeze(self.logits))  # scale to 0-1
            else:
                self.predict_score = tf.squeeze(self.logits)

        if self.scale_sentiment:
            print("Regressor scales sentiment score from 0 to 1")
            self.y_true = self.y * 0.2 - 0.1  # note: same as seq2sentiseq
        else:
            self.y_true = self.y
        self.loss = tf.losses.mean_squared_error(self.y_true, self.predict_score)
        self.mae_loss = tf.losses.absolute_difference(self.y_true, self.predict_score)

        if mode == constants.TRAIN:
            self.train_op = self.train()

        # only save regressor vars when dual training
        var_list = [var for var in tf.trainable_variables() if constants.REG_VAR_SCOPE in var.name]
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=10)  # Must in the end of model define

    def build_cell(self):
        cells = []
        for i in range(self.num_layers):
            cell = self.cell_fn(self.hidden_size)
            cell = rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            cells.append(cell)
        cell = rnn.MultiRNNCell(cells)
        return cell

    def attention(self, H):
        # Refer to "Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification"
        W = tf.get_variable("attention_w", shape=[self.hidden_size], initializer=self.initializer)

        M = tf.tanh(H)  # [batch_size, max_len, hidden_size]
        alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(M, [-1, self.hidden_size]),
                                                   tf.reshape(W, [-1, 1])),
                                         (-1, self.max_len)))  # [batch_size, seq_len]
        r = tf.matmul(tf.transpose(H, [0, 2, 1]),
                      tf.reshape(alpha, [-1, self.max_len, 1]))
        r = tf.squeeze(r)
        h_star = tf.tanh(r)  # [batch_size, hidden_size]
        return h_star

    def build_unirnn(self):
        with tf.variable_scope("UniRnn", initializer=self.initializer):
            cell = self.build_cell()
            initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(self.embedding, self.x)
            outputs, _ = tf.nn.dynamic_rnn(cell,
                                           inputs,
                                           sequence_length=self.sequence_length,
                                           initial_state=initial_state,
                                           dtype=tf.float32)
            output = self.attention(outputs)  # [batch_size, hidden_size]
        return output

    def build_birnn(self):
        with tf.variable_scope("BiRnn", initializer=self.initializer):
            fw_cell = self.build_cell()
            bw_cell = self.build_cell()
            initial_state_fw = fw_cell.zero_state(self.batch_size, tf.float32)
            initial_state_bw = bw_cell.zero_state(self.batch_size, tf.float32)
            inputs = tf.nn.embedding_lookup(self.embedding, self.x)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell,
                                                         bw_cell,
                                                         inputs,
                                                         sequence_length=self.sequence_length,
                                                         initial_state_fw=initial_state_fw,
                                                         initial_state_bw=initial_state_bw,
                                                         dtype=tf.float32)
            fw_outputs, bw_outputs = outputs
            outputs_sum = fw_outputs + bw_outputs  # [batch_size, max_len, hidden_size]
            output = self.attention(outputs_sum)  # [batch_size, hidden_size]
        return output

    def train(self):
        # trainable_varaibles = tf.trainable_variables()
        # train_op = optim.optimize(self.loss, self.params, trainable_varaibles)  # todo: change to normal optimizer
        # print("tf.train.GradientDescentOptimizer opennmt")
        # # train_op = tf.train.GradientDescentOptimizer(self.params.get("learning_rate", "0.001")).minimize(self.loss)
        # # print("tf.train.GradientDescentOptimizer me")

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.loss, tvars)
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
        train_op = optimizer.apply_gradients(zip(clipped_grads, tvars))

        return train_op
