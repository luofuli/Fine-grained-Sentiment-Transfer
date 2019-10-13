# -*- coding: utf-8 -*-
import sys, os
import re
import time
import glob
import tensorflow as tf
import numpy as np
from utils.data import load_dataset, load_paired_dataset
from utils.vocab import load_vocab
from utils.noise import add_noise
from seq2sentiseq.main import create_model as s2ss_create_model
from regressor.main import create_model as reg_create_model
from common_options import *
from cycle_options import load_cycle_arguments
from utils import constants
import argparse
from seq2sentiseq.main import inference
from utils.evaluator import BLEUEvaluator

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
safe_divide_constant = 0.000001


def get_tareget_sentiment(size, random=False):
    if random:
        s = np.random.randint(constants.MIN_SENT, constants.MAX_SENT + 1, size)
    else:
        s = np.repeat([constants.SENT_LIST], size, axis=0)
        s = s.reshape(-1)
    s = s.astype(np.float32)  # [1, 2, 3, 4, 5]
    return s


def process_mid_ids(ids_out, min_length, max_length, vocab_size):
    # ids_out has </s>
    seq_length = []
    for i in range(len(ids_out)):
        k = -1
        for j in range(max_length - 1):  # leave one place for ids_in_out
            assert max_length == len(ids_out[i])
            if ids_out[i][j] == constants.END_OF_SENTENCE_ID:
                k = j + 1
                break
        if k != -1:
            seq_length.append(k)
        else:
            # must have </s> in the end and leave one place for ids_in_out
            ids_out[i][max_length - 2] = constants.END_OF_SENTENCE_ID
            seq_length.append(max_length - 1)

    assert len(seq_length) == len(ids_out)

    def padded_to_min_length(ids_out, seq_length, min_length, vocab_size):
        append_count = 0
        batch_ids = ids_out.tolist()
        for i in range(len(batch_ids)):
            end_index = seq_length[i] - 1
            if not isinstance(batch_ids[i], list):
                batch_ids[i] = batch_ids[i].tolist()
            max_j = len(batch_ids[i])
            for k in range(end_index + 1, max_j):  # generated result may have some char after </s>, change to pad_id
                batch_ids[i][k] = constants.PADDING_ID
            for k in range(max_j, min_length + 1):  # to endure len(seq remove </s>)>= min_length
                batch_ids[i].append(constants.PADDING_ID)

            if end_index < min_length - 1:
                append_count += 1
                for j in range(end_index, min_length):
                    if j < len(batch_ids[i]):
                        batch_ids[i][j] = np.random.choice(vocab_size)
                    else:
                        batch_ids[i].append(np.random.choice(vocab_size))
                batch_ids[i][min_length] = constants.END_OF_SENTENCE_ID
                seq_length[i] = min_length + 1
        if append_count > 5:
            print("append_count:%d" % append_count)
        return np.array(batch_ids), seq_length

    def add_or_remove_tag(ids_out, seq_length, add_start=False, remove_end=False):
        batch_ids = ids_out.tolist()
        for i in range(len(batch_ids)):
            end_index = seq_length[i] - 1
            if add_start:
                end_index += 1
                batch_ids[i] = [constants.START_OF_SENTENCE_ID] + batch_ids[i]  # add <s>
            if remove_end:
                batch_ids[i][end_index] = constants.PADDING_ID  # remove </s>
                batch_ids[i] = batch_ids[i][:-1]  # shorten sequence
            len_ = len(batch_ids[i])
            if len_ < max_length:
                batch_ids[i].extend([constants.PADDING_ID] * (max_length - len_))
            else:
                batch_ids[i] = batch_ids[i][:max_length]
            assert len(batch_ids[i]) == max_length
        return np.array(batch_ids)

    ids_out, seq_length = padded_to_min_length(ids_out, seq_length, min_length=min_length, vocab_size=vocab_size)
    ids_in_out = add_or_remove_tag(ids_out, seq_length, add_start=True)
    ids_in = add_or_remove_tag(ids_out, seq_length, add_start=True, remove_end=True)
    ids = add_or_remove_tag(ids_out, seq_length, remove_end=True)
    ids_length = np.array(seq_length) - 1

    return ids, ids_in, ids_out, ids_in_out, ids_length


def main():
    args = load_cycle_arguments()
    dump_args_to_yaml(args, args.final_model_save_dir)
    print(args)

    reg_args = load_args_from_yaml(args.reg_model_save_dir)
    s2ss_args = load_args_from_yaml(args.s2ss_model_save_dir)
    # s2ss_args.seq2seq_model_save_dir = args.seq2seq_model_save_dir
    s2ss_args.RL_learning_rate = args.RL_learning_rate  # a smaller learning_rate for RL
    s2ss_args.MLE_learning_rate = args.MLE_learning_rate  # a smaller learning_rate for MLE
    s2ss_args.batch_size = args.batch_size  # a bigger batch_size for RL
    min_seq_len = args.min_seq_len
    max_seq_len = args.max_seq_len

    # === Load global vocab
    vocab, vocab_size = load_vocab(args.vocab_file)
    print("Vocabulary size: %s" % vocab_size)
    vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(
        args.vocab_file,  # target vocabulary file(each lines has a word)
        vocab_size=vocab_size - constants.NUM_OOV_BUCKETS,
        default_value=constants.UNKNOWN_TOKEN)

    bleu_evaluator = BLEUEvaluator()

    # === Create session
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=tf_config)  # limit gpu memory; don't pre-allocate memory; allocate as-needed

    # === Load dataset
    with tf.device("/cpu:0"):  # Input pipeline should always be place on the CPU.
        train_data_iterator = load_dataset(args.train_data, vocab, mode=constants.TRAIN, batch_size=args.batch_size,
                                           min_seq_len=min_seq_len, max_seq_len=max_seq_len)
        dev_data_iterator = load_dataset(args.dev_data, vocab, mode=constants.EVAL, batch_size=100,
                                         min_seq_len=min_seq_len, max_seq_len=max_seq_len)
        test_data_iterator = load_dataset(args.test_data, vocab, mode=constants.TEST, batch_size=100,
                                          min_seq_len=min_seq_len, max_seq_len=max_seq_len)
        paired_train_data_iterator = load_paired_dataset(args.pseudo_data, vocab, batch_size=args.batch_size,
                                                         min_seq_len=min_seq_len, max_seq_len=max_seq_len)

        train_data_next = train_data_iterator.get_next()  # to avoid high number of `Iterator.get_next()` calls
        dev_data_next = dev_data_iterator.get_next()
        test_data_next = test_data_iterator.get_next()
        paired_train_data_next = paired_train_data_iterator.get_next()

    # === Initialize and build Seq2SentiSeq model
    load_model = False if args.no_pretrain else True
    s2ss_train = s2ss_create_model(sess, s2ss_args, constants.TRAIN, vocab_size, load_pretrained_model=load_model)

    decode_type_before = s2ss_args.decode_type
    s2ss_args.decode_type = constants.GREEDY
    s2ss_greedy_infer = s2ss_create_model(sess, s2ss_args, constants.INFER, vocab_size, reuse=True)
    s2ss_args.decode_type = constants.RANDOM
    s2ss_random_infer = s2ss_create_model(sess, s2ss_args, constants.INFER, vocab_size, reuse=True)
    s2ss_args.decode_type = decode_type_before

    # === Load pre-trained sentiment regression model
    eval_reg = reg_create_model(sess, reg_args, vocab_size, mode=constants.EVAL, load_pretrained_model=True)

    print("Prepare for model saver")
    final_model_save_path = args.final_model_save_dir

    # === Start train
    n_batch = -1
    global_step = -1

    for i in range(args.n_epoch):
        print("Epoch:%s" % i)

        sess.run([train_data_iterator.initializer])
        sess.run([paired_train_data_iterator.initializer])

        senti_reward_all = {  # reward to measure the sentiment transformation of generated sequence
            "upper": [],  # reward of ground truth (existed sequence in train dataset)
            "lower": [],  # reward of baseline: random generated sequence
            "real": [],  # reward of real generated sequence
        }
        cont_reward_all = {  # reward to measure the content preservation of generated sequence
            "upper": [],  # reward of ground truth (existed sequence in train dataset)
            "lower": [],  # reward of baseline: random generated sequence
            "real": [],  # reward of real generated sequence
        }
        reward_all = []
        reward_expect_all = []  # reward expectation: r*p(y_k|x)

        while True:
            n_batch += 1
            global_step += 1
            if n_batch % args.eval_step == 0:
                print('\n================ N_batch / Global_step (%s / %s): Evaluate on test datasets ================\n'
                      % (n_batch, global_step))
                dst_fs = inference(s2ss_greedy_infer, sess=sess, args=s2ss_args, decoder_s=constants.SENT_LIST,
                                   src_test_iterator=test_data_iterator, src_test_next=test_data_next,
                                   vocab_rev=vocab_rev, result_dir=args.final_tsf_result_dir,
                                   step=global_step if args.save_each_step else global_step)
                t0 = time.time()
                bleu_scores = bleu_evaluator.score(args.reference, dst_fs[1], all_bleu=True)
                print("Test(Batch:%d)\tBLEU-1:%.3f\tBLEU-2:%.3f\tBLEU:%.3f\tCost time:%.2f" %
                      (n_batch, bleu_scores[1], bleu_scores[2], bleu_scores[0], time.time() - t0))

                # improve the diversity of generated sentences
                dst_fs = inference(s2ss_random_infer, sess=sess, args=s2ss_args, decoder_s=constants.SENT_LIST,
                                   src_test_iterator=test_data_iterator, src_test_next=test_data_next,
                                   vocab_rev=vocab_rev, result_dir=args.final_tsf_result_dir + '-sample',
                                   step=global_step if args.save_each_step else global_step)
                t0 = time.time()
                bleu_scores = bleu_evaluator.score(args.reference, dst_fs[1], all_bleu=True)
                print("Test(Batch:%d)\tBLEU-1:%.3f\tBLEU-2:%.3f\tBLEU:%.3f\tCost time:%.2f ===> Sampled results"
                      % (n_batch, bleu_scores[1], bleu_scores[2], bleu_scores[0], time.time() - t0))

            if n_batch % args.save_per_step == 0:
                print("Save model at dir:", final_model_save_path)
                s2ss_train.saver.save(sess, final_model_save_path, global_step=n_batch)

            try:
                t0 = time.time()
                src = sess.run(train_data_next)  # get real data!!
                batch_size = np.shape(src["ids"])[0]
                decode_width = s2ss_args.decode_width

                t0 = time.time()

                tile_src_ids = np.repeat(src["ids"], decode_width, axis=0)  # [batch_size*beam_size],
                tile_src_length = np.repeat(src['length'], decode_width, axis=0)
                tile_src_ids_in = np.repeat(src["ids_in"], decode_width, axis=0)
                tile_src_ids_out = np.repeat(src["ids_out"], decode_width, axis=0)
                tile_src_ids_in_out = np.repeat(src["ids_in_out"], decode_width, axis=0)
                tile_src_decoder_s = np.repeat(src["senti"], decode_width, axis=0)

                tile_tgt_decoder_s = get_tareget_sentiment(size=batch_size)
                tgt_decoder_s = get_tareget_sentiment(size=batch_size, random=True)

                t0 = time.time()

                # random
                random_predictions, log_probs = sess.run(
                    [s2ss_random_infer.predictions, s2ss_random_infer.log_probs],
                    feed_dict={s2ss_random_infer.encoder_input: tile_src_ids,
                               s2ss_random_infer.encoder_input_len: tile_src_length,
                               s2ss_random_infer.decoder_s: tile_tgt_decoder_s})

                mid_ids_log_prob = log_probs
                mid_ids, mid_ids_in, mid_ids_out, mid_ids_in_out, mid_ids_length = \
                    process_mid_ids(random_predictions, min_seq_len, max_seq_len, vocab_size)
                assert tile_src_length[0] == tile_src_length[decode_width - 1]

                # baseline
                greedy_predictions = sess.run(
                    s2ss_greedy_infer.predictions,
                    feed_dict={s2ss_greedy_infer.encoder_input: src['ids'],
                               s2ss_greedy_infer.encoder_input_len: src['length'],
                               s2ss_greedy_infer.decoder_s: tgt_decoder_s})

                mid_ids_bs, mid_ids_in_bs, mid_ids_out_bs, mid_ids_in_out_bs, mid_ids_length_bs = \
                    process_mid_ids(greedy_predictions, min_seq_len, max_seq_len, vocab_size)

                t0 = time.time()

                # == get reward from sentiment scorer/regressor
                def get_senti_reward(pred, gold):
                    if args.scale_sentiment:
                        gold = gold * 0.2 - 0.1  # todo: move this function to one file
                    reward_ = 1 / (np.fabs(pred - gold) + 1.0)
                    return reward_

                # real sentiment reward
                pred_senti_score = sess.run(eval_reg.predict_score,
                                            feed_dict={eval_reg.x: mid_ids,
                                                       eval_reg.sequence_length: mid_ids_length})
                senti_reward = get_senti_reward(pred_senti_score, tile_tgt_decoder_s)

                # upper bound of sentiment reward
                upper_pred_senti_score = sess.run(eval_reg.predict_score,
                                                  feed_dict={eval_reg.x: src["ids"],
                                                             eval_reg.sequence_length: src["length"]})
                upper_senti_reward = get_senti_reward(upper_pred_senti_score, src["senti"])

                # lower bound of sentiment reward
                lower_pred_senti_score = sess.run(eval_reg.predict_score,
                                                  feed_dict={
                                                      eval_reg.x: np.random.choice(vocab_size, np.shape(tile_src_ids)),
                                                      eval_reg.sequence_length: tile_src_length})
                lower_senti_reward = get_senti_reward(lower_pred_senti_score, tile_src_decoder_s)

                # == get reward from backward reconstruction
                feed_dict = {
                    s2ss_train.encoder_input: mid_ids,
                    s2ss_train.encoder_input_len: mid_ids_length,
                    s2ss_train.decoder_input: tile_src_ids_in,
                    s2ss_train.decoder_target: tile_src_ids_out,
                    s2ss_train.decoder_target_len: tile_src_length + 1,
                    s2ss_train.decoder_s: tile_src_decoder_s,
                }

                loss = sess.run(s2ss_train.loss_per_sequence, feed_dict=feed_dict)
                cont_reward = loss * (-1)  # bigger is better

                t0 = time.time()

                # get baseline content reward
                feed_dict = {
                    s2ss_train.encoder_input: mid_ids_bs,
                    s2ss_train.encoder_input_len: mid_ids_length_bs,
                    s2ss_train.decoder_input: src["ids_in"],
                    s2ss_train.decoder_target: src["ids_out"],
                    s2ss_train.decoder_target_len: src["length"] + 1,
                    s2ss_train.decoder_s: src["senti"],
                }
                loss_bs = sess.run(s2ss_train.loss_per_sequence, feed_dict=feed_dict)
                cont_reward_bs = loss_bs * (-1)  # baseline content reward

                # get lower bound of content reward
                feed_dict = {
                    s2ss_train.encoder_input: np.random.choice(vocab_size, np.shape(mid_ids)),
                    s2ss_train.encoder_input_len: mid_ids_length,
                    s2ss_train.decoder_input: np.random.choice(vocab_size, np.shape(tile_src_ids_in)),
                    s2ss_train.decoder_target: np.random.choice(vocab_size, np.shape(tile_src_ids_out)),
                    s2ss_train.decoder_target_len: tile_src_length + 1,
                    s2ss_train.decoder_s: tile_src_decoder_s,
                }
                lower_loss = sess.run(s2ss_train.loss_per_sequence, feed_dict=feed_dict)
                lower_cont_reward = lower_loss * (-1)  # bigger is better

                def norm(x):
                    x = np.array(x)
                    x = (x - x.mean()) / (x.std() + 1e-6)  # safe divide
                    # x = x - x.min()  # to make x > 0
                    return x

                def sigmoid(x, x_trans=0.0, x_scale=1.0, max_y=1, do_norm=False):
                    value = max_y / (1 + np.exp(-(x - x_trans) * x_scale))
                    if do_norm:
                        value = norm(value)
                    return value

                def norm_s2ss_reward(x, baseline=None, scale=False, norm=False):
                    x = np.reshape(x, (batch_size, -1))  # x in [-16, 0]
                    dim1 = np.shape(x)[1]

                    if baseline is not None:
                        x_baseline = baseline  # [batch_size]
                    else:
                        x_baseline = np.mean(x, axis=1)  # [batch_size]
                    x_baseline = np.repeat(x_baseline, dim1)  # [batch_size*dim1]
                    x_baseline = np.reshape(x_baseline, (batch_size, dim1))

                    x_norm = x - x_baseline

                    if scale:
                        x_norm = sigmoid(x_norm)
                    if norm:
                        x_norm = 2 * x_norm - 1  # new x_norm in [-1, 1]
                    return x_norm.reshape(-1)

                if args.use_baseline:
                    if global_step < 1:  # only print at first 10 steps
                        print('%%% use_baseline')
                    cont_reward = norm_s2ss_reward(cont_reward, baseline=cont_reward_bs, scale=True)
                    lower_cont_reward = norm_s2ss_reward(lower_cont_reward, baseline=cont_reward_bs, scale=True)

                elif args.scale_cont_reward:
                    if global_step < 1:  # only print at first 1 steps
                        print('%%% scale_cont_reward')
                    cont_reward = sigmoid(cont_reward, x_trans=-3)  # [-6, -2] => [0.1, 0.78]
                    lower_cont_reward = sigmoid(lower_cont_reward, x_trans=-3)

                if args.scale_senti_reward:
                    if global_step < 1:  # only print at first 1 steps
                        print('%%% scale_senti_reward')
                    senti_reward = sigmoid(senti_reward, x_trans=-0.8, x_scale=15)  # [0.6, 1.0] => [0.04, 0.95]
                    lower_senti_reward = sigmoid(lower_senti_reward, x_trans=-0.8, x_scale=15)
                    upper_senti_reward = sigmoid(upper_senti_reward,  x_trans=-0.8, x_scale=15)

                cont_reward_all["lower"].extend(lower_cont_reward)
                cont_reward_all["real"].extend(cont_reward)

                senti_reward_all["upper"].extend(upper_senti_reward)
                senti_reward_all["lower"].extend(lower_senti_reward)
                senti_reward_all["real"].extend(senti_reward)

                senti_reward += safe_divide_constant
                cont_reward += safe_divide_constant

                if args.increase_beta:
                    beta = min(1, 0.1 * global_step / args.increase_step)
                else:
                    beta = 1

                reward_merge_type = 'H(sentiment, content), beta=%.2f' % beta  # enlarge the influence of senti_reward
                reward = (1 + beta * beta) * senti_reward * cont_reward / (beta * beta * senti_reward + cont_reward)

                reward_all.extend(reward)
                reward_expect_all.extend(reward * np.exp(mid_ids_log_prob))

                # policy gradient training
                if not args.no_RL:
                    feed_dict = {
                        s2ss_train.encoder_input: tile_src_ids,
                        s2ss_train.encoder_input_len: tile_src_length,
                        s2ss_train.decoder_input: mid_ids_in,
                        s2ss_train.decoder_target: mid_ids_out,
                        s2ss_train.decoder_target_len: mid_ids_length + 1,
                        s2ss_train.decoder_s: tile_tgt_decoder_s,
                        s2ss_train.reward: reward
                    }
                    sess.run([s2ss_train.rl_loss, s2ss_train.retrain_op], feed_dict=feed_dict)

                # Teacher forcing data types:
                #  1. back translation data (greedy decode)
                #  2. back translation data (random decode)
                #  3. back translation noise data
                #  4. pseudo data
                #  5. same data (x->x)
                #  6. same_noise (x'->x)

                if "back_trans" in args.teacher_forcing:
                    if args.MLE_decay:
                        if args.MLE_decay_type == "linear":
                            gap = min(10, 2 + global_step / args.MLE_decay_steps)  # 10 after 1 epoch
                        else:
                            gap = min(5, int(1 / np.power(args.MLE_decay_rate, global_step / args.MLE_decay_steps)))
                    else:
                        gap = 1
                    if n_batch % gap == 0:
                        if global_step < 1 :
                            print('$$$Update B use back-translated data (Update gap:%s)' % gap)
                        # Update Seq2SentiSeq with previous model generated data  # senti-, bleu+
                        feed_dict = {
                            s2ss_train.encoder_input: mid_ids_bs,
                            s2ss_train.encoder_input_len: mid_ids_length_bs,
                            s2ss_train.decoder_input: src["ids_in"],
                            s2ss_train.decoder_target: src["ids_out"],
                            s2ss_train.decoder_target_len: src["length"] + 1,
                            s2ss_train.decoder_s: src["senti"],
                        }
                        sess.run([s2ss_train.loss, s2ss_train.train_op], feed_dict=feed_dict)

                if "back_trans_random" in args.teacher_forcing:
                    if args.MLE_decay:
                        if args.MLE_decay_type == "linear":
                            gap = min(10, 2 + global_step / args.MLE_decay_steps)  # 10 after 1 epoch
                        else:
                            gap = min(5, int(1 / np.power(args.MLE_decay_rate, global_step / args.MLE_decay_steps)))
                    else:
                        gap = 1
                    if n_batch % gap == 0:
                        if global_step < 1 :
                            print('$$$Update B use back_trans_random data (Update gap:%s)' % gap)
                        # Update Seq2SentiSeq with previous model generated data with noise
                        feed_dict = {
                            s2ss_train.encoder_input: mid_ids,
                            s2ss_train.encoder_input_len: mid_ids_length,
                            s2ss_train.decoder_input: tile_src_ids_in,
                            s2ss_train.decoder_target: tile_src_ids_out,
                            s2ss_train.decoder_target_len: tile_src_length + 1,
                            s2ss_train.decoder_s: tile_src_decoder_s,
                        }
                        sess.run([s2ss_train.loss, s2ss_train.train_op], feed_dict=feed_dict)

                if "back_trans_noise" in args.teacher_forcing:
                    if args.MLE_decay:
                        if args.MLE_decay_type == "linear":
                            gap = min(10, 2 + global_step / args.MLE_decay_steps)  # 10 after 1 epoch
                        else:
                            gap = min(5, int(1 / np.power(args.MLE_decay_rate, global_step / args.MLE_decay_steps)))
                    else:
                        gap = 1
                    if n_batch % gap == 0:
                        if global_step < 1 :
                            print('$$$Update B use back_trans_noise data (Update gap:%s)' % gap)
                        # Update Seq2SentiSeq with previous model generated data with noise
                        noise_ids, noise_ids_length = add_noise(mid_ids_bs, mid_ids_length_bs)
                        feed_dict = {
                            s2ss_train.encoder_input: noise_ids,
                            s2ss_train.encoder_input_len: noise_ids_length,
                            s2ss_train.decoder_input: src["ids_in"],
                            s2ss_train.decoder_target: src["ids_out"],
                            s2ss_train.decoder_target_len: src["length"] + 1,
                            s2ss_train.decoder_s: src["senti"],
                        }
                        sess.run([s2ss_train.loss, s2ss_train.train_op], feed_dict=feed_dict)

                if "pseudo_data" in args.teacher_forcing:  # balance
                    if args.MLE_decay:
                        if args.MLE_decay_type == "linear":
                            gap = min(10, 3 + global_step / args.MLE_decay_steps)  # 10 after 1 epoch
                        else:
                            gap = min(100, int(3 / np.power(args.MLE_decay_rate, global_step / args.MLE_decay_steps)))
                    else:
                        gap = 3
                    if n_batch % gap == 0:
                        if global_step < 1 :
                            print('$$$Update use pseudo data (Update gap:%s)' % gap)
                        data = sess.run(paired_train_data_next)  # get real data!!
                        feed_dict = {
                            s2ss_train.encoder_input: data["source_ids"],
                            s2ss_train.encoder_input_len: data["source_length"],
                            s2ss_train.decoder_input: data["target_ids_in"],
                            s2ss_train.decoder_target: data["target_ids_out"],
                            s2ss_train.decoder_target_len: data["target_length"] + 1,
                            s2ss_train.decoder_s: data["target_senti"]
                        }
                        sess.run([s2ss_train.loss, s2ss_train.train_op], feed_dict=feed_dict)

                if "same" in args.teacher_forcing:
                    if args.same_decay:
                        if args.same_decay_type == "linear":
                            gap = min(8, 2 + global_step / args.same_decay_steps)  # 10 after 1 epoch
                        else:
                            gap = min(10, int(2 / np.power(args.same_decay_rate, global_step / args.same_decay_rate)))
                    else:
                        gap = 2
                    if n_batch % gap == 0:
                        print('$$$Update use same data (Update gap:%s)' % gap)
                        # Update Seq2SentiSeq with target output  # senti-, bleu+
                        feed_dict = {
                            s2ss_train.encoder_input: src["ids"],
                            s2ss_train.encoder_input_len: src["length"],
                            s2ss_train.decoder_input: src["ids_in"],
                            s2ss_train.decoder_target: src["ids_out"],
                            s2ss_train.decoder_target_len: src["length"] + 1,
                            s2ss_train.decoder_s: src["senti"]
                        }
                        sess.run([s2ss_train.loss, s2ss_train.train_op], feed_dict=feed_dict)

                if "same_noise" in args.teacher_forcing:
                    if args.same_decay:
                        if args.same_decay_type == "linear":
                            gap = min(8, 2 + global_step / args.same_decay_steps)  # 10 after 1 epoch
                        else:
                            gap = min(10, int(2 / np.power(args.same_decay_rate, global_step / args.same_decay_rate)))
                    else:
                        gap = 2
                    if n_batch % gap == 0:
                        print('$$$Update use same_noise data (Update gap:%s)' % gap)
                        noise_ids, noise_ids_length = add_noise(src["ids"], src["length"])
                        feed_dict = {
                            s2ss_train.encoder_input: noise_ids,
                            s2ss_train.encoder_input_len: noise_ids_length,
                            s2ss_train.decoder_input: src["ids_in"],
                            s2ss_train.decoder_target: src["ids_out"],
                            s2ss_train.decoder_target_len: src["length"] + 1,
                            s2ss_train.decoder_s: src["senti"]
                        }
                        sess.run([s2ss_train.loss, s2ss_train.train_op], feed_dict=feed_dict)
            except tf.errors.OutOfRangeError:  # next epoch
                print("Train---Total N batch:{}\tCost time:{}".format(n_batch, time.time() - t0))
                n_batch = -1
                break


if __name__ == "__main__":
    main()
