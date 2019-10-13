# -*- coding: utf-8 -*-

import math
import tensorflow as tf
import numpy as np
from utils import constants
from tensorflow.contrib.rnn import LSTMCell, GRUCell, LSTMStateTuple, DropoutWrapper

dynamic_v = False


class Seq2SentiSeq(object):
    def __init__(self, mode, cell_type, num_hidden, embedding_seman_size, embedding_senti_size, vocab_size, max_seq_len,
                 decode_type, mle_learning_rate, rl_learning_rate, softmax_temperature, grad_clip, scale_sentiment):
        self.mode = mode
        self.cell_type = cell_type
        self.vocab_size = vocab_size
        self.embedding_seman_size = embedding_seman_size
        self.embedding_senti_size = embedding_senti_size
        self.num_hidden = num_hidden
        self.max_seq_len = max_seq_len
        self.grad_clip = grad_clip
        self.sample = True if decode_type == constants.RANDOM else False
        self.keep_prob = 0.5 if mode == constants.TRAIN else 1.0
        self.MLE_learning_rate = mle_learning_rate
        self.RL_learning_rate = rl_learning_rate
        self.softmax_temperature = softmax_temperature
        self.scale_sentiment = scale_sentiment

        print('self.MLE_learning_rate', self.MLE_learning_rate)
        print('self.RL_learning_rate', self.RL_learning_rate)

        self._check_args()

        if self.cell_type == 'lstm':
            self.cell_fn = lambda x: DropoutWrapper(LSTMCell(x, state_is_tuple=True), output_keep_prob=self.keep_prob)
        elif self.cell_type == 'gru':
            self.cell_fn = lambda x: DropoutWrapper(GRUCell(x), output_keep_prob=self.keep_prob)
        self._create_placeholders()
        self._create_variable()
        self._create_network()

    def _check_args(self):
        if self.cell_type not in ['lstm', 'gru']:
            raise ValueError("This cell type is not supported.")

    def _create_placeholders(self):
        with tf.variable_scope('placeholders'):
            # batch_size * sentence_num * word_num
            self.encoder_input = tf.placeholder(tf.int32, [None, self.max_seq_len], name='encoder_input')
            self.encoder_input_len = tf.placeholder(tf.int32, [None, ], name='encoder_sequence_lengths')

            self.decoder_input = tf.placeholder(tf.int32, [None, self.max_seq_len], name="decoder_input")
            self.decoder_target = tf.placeholder(tf.int32, [None, self.max_seq_len], name="decoder_target")
            self.decoder_target_len = tf.placeholder(tf.float32, [None], name="decoder_target_len")
            self.decoder_s = tf.placeholder(tf.float32, [None], name="decoder_s")

            self.reward = tf.placeholder(tf.float32, shape=(None,), name="reward")

    def _create_variable(self):

        with tf.variable_scope('embeddings', initializer=tf.random_uniform_initializer(-0.08, 0.08)):
            self.embedding_seman = tf.get_variable("embedding_seman", [self.vocab_size, self.embedding_seman_size])
            # initializer_senti = tf.random_normal_initializer(mean=0.0, stddev=0.01)  # better for sentiment embedding?
            self.embedding_senti = tf.get_variable("embedding_senti", [self.vocab_size, self.embedding_senti_size])

        with tf.variable_scope('decoder_output_projection', initializer=tf.contrib.layers.xavier_initializer()):
            self.seman_w = tf.get_variable("seman_w", [self.num_hidden, self.vocab_size])
            self.seman_b = tf.get_variable("seman_b", [self.vocab_size])
            self.senti_u = tf.get_variable("senti_u", [self.num_hidden, self.embedding_senti_size])

            self.sum_weight = tf.constant(0.5)
            self.v1 = tf.get_variable("v1", [self.vocab_size])
            self.v2 = tf.get_variable("v2", [self.vocab_size])

    def _create_encoder(self, encoder_input, encoder_input_len):
        with tf.variable_scope('encoder'):
            encoder_input_embedded = tf.nn.embedding_lookup(self.embedding_seman, encoder_input)
            cell_fw = self.cell_fn(self.num_hidden)
            cell_bw = self.cell_fn(self.num_hidden)
            (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                cell_bw=cell_bw,
                                                inputs=encoder_input_embedded,
                                                sequence_length=encoder_input_len,
                                                dtype=tf.float32)
            encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

        return encoder_outputs, encoder_fw_final_state  # encoder_bw_final_state is random initial

    def _create_decoder(self, encoder_state, encoder_output, decoder_input, decoder_s, sample=False):
        with tf.variable_scope('decoder', initializer=tf.contrib.layers.xavier_initializer()):

            decoder_seman_embedded = tf.nn.embedding_lookup(self.embedding_seman, decoder_input)
            decoder_senti_embedded = tf.nn.embedding_lookup(self.embedding_senti, decoder_input)
            decoder_embedded = tf.concat([decoder_seman_embedded, decoder_senti_embedded], axis=-1)  # [B, L, 2D]
            decoder_embedded_list = tf.unstack(decoder_embedded, axis=1)  # L * [B, 2D]

            decoder_s = tf.expand_dims(decoder_s, -1)  # [B] => [B, 1]

            cell = self.cell_fn(self.num_hidden)

            if self.mode == constants.TRAIN:  # Train

                decoder_outputs, decoder_state = tf.contrib.legacy_seq2seq.attention_decoder(
                    decoder_inputs=decoder_embedded_list,
                    initial_state=encoder_state,
                    attention_states=encoder_output,
                    cell=cell,
                    initial_state_attention=True)
                seman_logits = [tf.nn.xw_plus_b(x, self.seman_w, self.seman_b) for x in decoder_outputs]  # L*[B, V]
                seman_logits = tf.stack(seman_logits, 1)  # L*[B, V] => [B, L, V]
                seman_probs = tf.nn.softmax(seman_logits)

                decoder_outputs_senti = [tf.matmul(tf.matmul(x, self.senti_u), self.embedding_senti, transpose_b=True)
                                         for x in decoder_outputs]  # L*[B, V]
                senti_value = tf.stack(decoder_outputs_senti, 1)  # L*[B, V] => [B, L, V]
                senti_value = tf.nn.sigmoid(senti_value)

                gaussian_s = tf.tile(decoder_s, [1, self.max_seq_len * self.vocab_size])  # [B, 1] => [B, L*V]
                gaussian_s = tf.reshape(gaussian_s, [-1, self.max_seq_len, self.vocab_size])  # [B, L*V] => [B, L, V]
                factor = tf.multiply((gaussian_s - senti_value), (gaussian_s - senti_value)) / 2
                gaussian_p = tf.exp(-factor) * math.pow(2 * math.pi, -0.5)
                senti_logits = gaussian_p / self.softmax_temperature
                senti_probs = tf.nn.softmax(senti_logits)

                if not dynamic_v:
                    probs = self.sum_weight * seman_probs + (1 - self.sum_weight) * senti_probs
                    self.v = self.sum_weight
                else:
                    v1 = tf.reshape(self.v1, [1, 1, self.vocab_size])
                    v2 = tf.reshape(self.v2, [1, 1, self.vocab_size])
                    v = tf.reduce_sum(v1 * seman_probs + v2 * senti_probs, axis=-1)  # [B, L]
                    v = tf.expand_dims(tf.nn.sigmoid(v), -1)  # [B, L, 1]
                    self.v = v
                    probs = v * seman_probs + (1 - v) * senti_probs

            else:  # Inference
                def _extract_argmax_and_embed(seman_embedding, senti_embedding, seman_projection, senti_u,
                                              softmax_temperature, sum_weight=None, vs=None, sample=False):
                    def loop_function(prev, _):
                        prev_seman_logits = tf.nn.xw_plus_b(prev, seman_projection[0], seman_projection[1])
                        prev_seman_probs = tf.nn.softmax(prev_seman_logits)  # [B, V]

                        prev_senti = tf.matmul(tf.matmul(prev, senti_u), senti_embedding, transpose_b=True)
                        senti_value = tf.nn.sigmoid(prev_senti)  # [B, V]
                        gaussian_s = tf.tile(decoder_s, [1, self.vocab_size])  # [B, 1] => [B, V]
                        factor = tf.multiply((gaussian_s - senti_value), (gaussian_s - senti_value)) / 2
                        gaussian_p = tf.exp(-factor) * math.pow(2 * math.pi, -0.5)
                        prev_senti_logits = gaussian_p / softmax_temperature
                        prev_senti_probs = tf.nn.softmax(prev_senti_logits)

                        if not dynamic_v:
                            prev_probs = sum_weight * prev_seman_probs + (1 - sum_weight) * prev_senti_probs
                        else:
                            v1 = tf.expand_dims(vs[0], 0)  # [1, B]
                            v2 = tf.expand_dims(vs[1], 0)
                            v = tf.reduce_sum(v1 * prev_seman_probs + v2 * prev_senti_probs, axis=-1)  # [B]
                            v = tf.expand_dims(tf.nn.sigmoid(v), -1)  # [B, 1]
                            prev_probs = v * prev_seman_probs + (1 - v) * prev_senti_probs

                        if sample:  # Sample from the full output distribution
                            symbol = tf.distributions.Categorical(probs=prev_probs).sample()
                        else:  # Sample best prediction
                            symbol = tf.argmax(prev_probs, 1)

                        seman_embed = tf.nn.embedding_lookup(seman_embedding, symbol)
                        senti_embed = tf.nn.embedding_lookup(senti_embedding, symbol)
                        embed = tf.concat([seman_embed, senti_embed], axis=-1)

                        return embed

                    return loop_function

                loop_function_predict = _extract_argmax_and_embed(self.embedding_seman,
                                                                  self.embedding_senti,
                                                                  seman_projection=(self.seman_w, self.seman_b),
                                                                  senti_u=self.senti_u,
                                                                  sum_weight=self.sum_weight,
                                                                  vs=(self.v1, self.v2),
                                                                  softmax_temperature=self.softmax_temperature,
                                                                  sample=sample)

                decoder_pred_hiddens, _ = tf.contrib.legacy_seq2seq.attention_decoder(
                    decoder_inputs=decoder_embedded_list,
                    initial_state=encoder_state,
                    cell=cell,
                    attention_states=encoder_output,
                    loop_function=loop_function_predict,
                    initial_state_attention=True)

                seman_logits = [tf.nn.xw_plus_b(x, self.seman_w, self.seman_b) for x in decoder_pred_hiddens]
                seman_logits = tf.stack(seman_logits, 1)  # L*[B, V] => [B, L, V]
                seman_probs = tf.nn.softmax(seman_logits)

                decoder_outputs_senti = [tf.matmul(tf.matmul(x, self.senti_u), self.embedding_senti, transpose_b=True)
                                         for x in decoder_pred_hiddens]  # L*[B, V]
                senti_value = tf.stack(decoder_outputs_senti, 1)  # L*[B, V] => [B, L, V]
                senti_value = tf.nn.sigmoid(senti_value)

                gaussian_s = tf.tile(decoder_s, [1, self.max_seq_len * self.vocab_size])  # [B, 1] => [B, L*V]
                gaussian_s = tf.reshape(gaussian_s, [-1, self.max_seq_len, self.vocab_size])  # [B, L*V] => [B, L, V]
                factor = tf.multiply((gaussian_s - senti_value), (gaussian_s - senti_value)) / 2
                gaussian_p = tf.exp(-factor) * math.pow(2 * math.pi, -0.5)
                senti_logits = gaussian_p / self.softmax_temperature
                senti_probs = tf.nn.softmax(senti_logits)

                if not dynamic_v:
                    probs = self.sum_weight * seman_probs + (1 - self.sum_weight) * senti_probs
                    self.v = self.sum_weight
                else:
                    v1 = tf.reshape(self.v1, [1, 1, self.vocab_size])
                    v2 = tf.reshape(self.v2, [1, 1, self.vocab_size])
                    v = tf.reduce_sum(v1 * seman_probs + v2 * senti_probs, axis=-1)  # [B, L]
                    v = tf.expand_dims(tf.nn.sigmoid(v), -1)  # [B, L, 1]
                    self.v = v
                    probs = v * seman_probs + (1 - v) * senti_probs

        return probs

    def _create_network(self):

        self.encoder_outputs, self.encoder_final_state = self._create_encoder(self.encoder_input,
                                                                              self.encoder_input_len)

        if self.mode == constants.TRAIN:
            decoder_input = self.decoder_input
        else:
            # Only input <start> token
            batch_size = tf.shape(self.encoder_input)[0]
            decoder_input = tf.tile([[constants.START_OF_SENTENCE_ID]], [batch_size, self.max_seq_len])

        if self.scale_sentiment:
            print("Seq2SentiSeq scales sentiment score from 0 to 1")
            self.decoder_s_real = self.decoder_s * 0.2 - 0.1  # important
        else:
            self.decoder_s_real = self.decoder_s

        self.probs = self._create_decoder(self.encoder_final_state,
                                          self.encoder_outputs,
                                          decoder_input,
                                          self.decoder_s_real,
                                          sample=self.sample)

        if self.mode == constants.TRAIN:
            self.decoder_weights = tf.sequence_mask(self.decoder_target_len, maxlen=self.max_seq_len, dtype=tf.float32)

            logits = tf.log(self.probs + 1e-7)
            self.loss_per_sequence = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                                                      targets=self.decoder_target,
                                                                      weights=self.decoder_weights,
                                                                      average_across_timesteps=True,
                                                                      average_across_batch=False)
            # calculate cross-entropy loss
            self.loss = tf.reduce_mean(self.loss_per_sequence)
            self.train_op = self.get_train_op(self.loss, self.MLE_learning_rate)

            # calculate reinforcement loss
            rl_loss_per_sequence = self.loss_per_sequence * self.reward
            self.rl_loss = tf.reduce_mean(rl_loss_per_sequence)
            self.retrain_op = self.get_train_op(self.rl_loss, self.RL_learning_rate)
        else:
            probs_per_token = tf.reduce_max(self.probs, axis=-1)  # [B, L, V] => [B, L]
            log_probs_per_token = tf.log(probs_per_token + 1e-7)
            self.log_probs = tf.reduce_mean(log_probs_per_token, axis=1)
            self.predictions = tf.argmax(self.probs, axis=-1)

        # only save vars of CNMT when dual training
        var_list = [var for var in tf.global_variables() if constants.S2S_VAR_SCOPE in var.name]
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=10)  # Must in the end of model define

    def get_train_op(self, loss, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
        train_op = optimizer.apply_gradients(zip(clipped_grads, tvars))
        return train_op
