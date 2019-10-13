# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import sys, os

sys.path.append("..")

from utils.data import load_dataset, load_paired_dataset
from utils.vocab import build_vocab_from_file, load_vocab
from model import Seq2SentiSeq
from utils import constants
from options import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def create_model(sess, params, mode, vocab_size, load_pretrained_model=False, model_save_dir=None, reuse=None):
    print("-- Create Seq2SentiSeq model (mode: %s) ---" % mode)

    with tf.variable_scope(constants.S2S_VAR_SCOPE, reuse=reuse):
        sess.run(tf.tables_initializer())
        model = Seq2SentiSeq(mode=mode,
                             cell_type=params.cell_type,
                             num_hidden=params.num_hidden,
                             embedding_seman_size=params.semantic_embedding_size,
                             embedding_senti_size=params.sentiment_embedding_size,
                             vocab_size=vocab_size,
                             max_seq_len=params.max_seq_len,
                             decode_type=params.decode_type,
                             mle_learning_rate=params.MLE_learning_rate,
                             rl_learning_rate=params.RL_learning_rate,
                             softmax_temperature=params.softmax_temperature,
                             grad_clip=params.clip_gradients,
                             scale_sentiment=params.scale_sentiment)

    assert load_pretrained_model in [True, False]

    if load_pretrained_model is True:
        if model_save_dir is None:
            model_save_dir = params.s2ss_model_save_dir
        try:
            print("Loading Seq2SentiSeq model from", model_save_dir)
            model.saver.restore(sess, model_save_dir)
        except Exception as e:
            print("Error! Loading Seq2SentiSeq model from", model_save_dir)
            print("Again! Loading Seq2SentiSeq model from", tf.train.latest_checkpoint(model_save_dir))
            model.saver.restore(sess, tf.train.latest_checkpoint(model_save_dir))
    else:
        if reuse is None:
            print("Creating Seq2SentiSeq model with fresh parameters.")
            sess.run(tf.global_variables_initializer())
        else:
            print("Reuse Seq2SentiSeq parameters.")
    return model


def train(model):
    best = {
        "loss": 100.0,
        "step": 0,
    }

    global_step = -1
    for i in range(args.n_epoch):
        print("Epoch:%d" % i)
        sess.run(train_iterator.initializer)
        n_batch = -1
        t0 = time.time()
        while True:
            try:
                n_batch += 1
                global_step += 1

                src = sess.run(train_next_op)  # get real data!!
                feed_dict = {model.encoder_input: src["source_ids"],
                             model.encoder_input_len: src["source_length"],
                             model.decoder_input: src["target_ids_in"],
                             model.decoder_target: src["target_ids_out"],
                             model.decoder_target_len: src["target_length"] + 1,
                             model.decoder_s: src["target_senti"]}
                ops = [model.loss, model.v, model.decoder_s_real, model.train_op]
                res = sess.run(ops, feed_dict=feed_dict)

                if n_batch % 100 == 0:
                    print('v_min:%.3f, v_max:%.3f' % (np.min(res[1]), np.max(res[1])))
                    print('decoder_s_real: min:%.3f, max:%.3f' % (np.min(res[2]), np.max(res[2])))
                    print("Epoch/n_batch/global_step:%d/%d/%s\tTrain_loss:%.3f\tTime:%d" % (
                        i, n_batch, global_step, res[0], time.time() - t0))
                    model.saver.save(sess, args.s2ss_model_save_dir, global_step=n_batch)
                    print("Save model to: %s" % args.s2ss_model_save_dir)

            except tf.errors.OutOfRangeError:  # next epoch
                print("Train---Total N batch:{}".format(n_batch))
                break
            except tf.errors.InvalidArgumentError as e:
                print(e)
                continue


def inference(model, sess, args, src_test_iterator, src_test_next, vocab_rev, decoder_s, result_dir=None, step=None):
    sess.run([src_test_iterator.initializer])

    n_batch = 0
    t0 = time.time()
    if result_dir is not None:
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        src_test_file = args.test_data.split("/")[-1].replace('.txt', '')
        step = str(step) + "_" if step is not None else ''
        result_save_path1 = result_dir + "/" + step + src_test_file + '_has_source.tsf'
        result_save_path2 = result_dir + "/" + step + src_test_file + '.tsf'
        print("Result save path:" + result_save_path1 + ", " + result_save_path2)
        dst_f1 = open(result_save_path1, "w")
        dst_f2 = open(result_save_path2, "w")

    log_probs = []
    while True:
        try:
            n_batch += 1
            src = sess.run(src_test_next)  # get real data!!

            decode_width = len(decoder_s)

            batch_size = len(src["ids"])
            tile_src_ids = np.repeat(src["ids"], decode_width, axis=0)  # [batch_size*decode_width],
            tile_src_length = np.repeat(src['length'], decode_width, axis=0)
            tile_tgt_decoder_s = np.repeat([decoder_s], batch_size, axis=0)
            tile_tgt_decoder_s = tile_tgt_decoder_s.reshape(-1).astype(np.float32)  # [decode_width*batch_size]

            assert tile_tgt_decoder_s[0] == tile_tgt_decoder_s[decode_width]

            feed_dict = {model.encoder_input: tile_src_ids,
                         model.encoder_input_len: tile_src_length,
                         model.decoder_s: tile_tgt_decoder_s}
            t0 = time.time()
            predictions, log_prob = sess.run([model.predictions, model.log_probs], feed_dict=feed_dict)

            log_probs.extend(log_prob)

            t0 = time.time()
            if result_dir is not None:
                src_tokens = sess.run(vocab_rev.lookup(tf.cast(src["ids"], tf.int64)))
                pred_tokens = sess.run(vocab_rev.lookup(tf.cast(predictions, tf.int64)))
                pred_tokens = np.reshape(pred_tokens, [batch_size, decode_width, -1])
                for i in range(batch_size):
                    src_tokens_ = src_tokens[i][:src["length"][i]]
                    src_sent = " ".join(src_tokens_)
                    for j in range(decode_width):
                        pred_token_ = []
                        for s in pred_tokens[i][j]:
                            if s == constants.END_OF_SENTENCE_TOKEN:
                                break  # Ignore </s>.
                            else:
                                pred_token_.append(s)
                        # max_seq_len - 2 => spare two words for data loader when evaluating in cycle_training
                        pred_sent = ' '.join(pred_token_[:args.max_seq_len-2])
                        dst_f1.write("%s\t%s\t%s\n" % (src_sent, pred_sent, tile_tgt_decoder_s[i * decode_width + j]))
                        dst_f2.write("%s\n" % pred_sent)

        except tf.errors.OutOfRangeError as e:  # next epoch
            # print(e)
            print("INFERENCE---Total N batch:{}\tLog prob:{}\tCost time:{}".format(
                n_batch, np.mean(log_probs), time.time() - t0))
            break

    if result_dir is not None:
        dst_f1.close()
        dst_f2.close()
    return [result_save_path1, result_save_path2]


if __name__ == "__main__":
    args = load_s2ss_arguments()

    # Step 1: build vocab and load data
    print("Sharing vocabulary")
    if not os.path.isfile(args.vocab_file):
        build_vocab_from_file(args.train_data, args.vocab_file)
        print("Build vocabulary")
    vocab, vocab_size = load_vocab(args.vocab_file)
    print('Vocabulary size:%s' % vocab_size)

    vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(
        args.vocab_file,  # target vocabulary file(each lines has a word)
        vocab_size=vocab_size - constants.NUM_OOV_BUCKETS,
        default_value=constants.UNKNOWN_TOKEN)

    with tf.device("/cpu:0"):  # Input pipeline should always be place on the CPU.

        print("args.pseudo_data:", args.pseudo_data)

        if args.mode == "train":
            train_iterator = load_paired_dataset(args.pseudo_data, vocab, batch_size=args.batch_size,
                                                 min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len)
            train_next_op = train_iterator.get_next()
        else:
            src_test_iterator = load_dataset(args.test_data, vocab, mode=constants.INFER,
                                             min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len)
            src_test_next_op = src_test_iterator.get_next()

    # Step 2: create session
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=tf_config)  # limit gpu memory; don"t pre-allocate memory; allocate as-needed

    # Step 3: train model
    if args.mode == "train":
        # Prepare for model saver
        print("Prepare for model saver")
        print("Model save dir:", args.s2ss_model_save_dir)
        if not os.path.exists(args.s2ss_model_save_dir):
            os.makedirs(args.s2ss_model_save_dir)
        dump_args_to_yaml(args, args.s2ss_model_save_dir)

        # Initial and build model
        train_model = create_model(sess, args, constants.TRAIN, vocab_size)
        # infer_model = create_model(sess, args, vocab_size, reuse=True)

        train(train_model)

    elif args.mode == "inference":
        infer_model = create_model(sess, args, constants.INFER, vocab_size, load_pretrained_model=True)
        inference(infer_model, sess=sess, args=args, src_test_iterator=src_test_iterator, decoder_s=constants.SENT_LIST,
                  src_test_next=src_test_iterator.get_next(), vocab_rev=vocab_rev, result_dir=args.tsf_result_dir)

    elif args.mode == "final_inference":  # todo: check data load
        print("Prepare for model saver")
        final_model_save_path = args.final_model_save_dir

        args.decode_type = constants.GREEDY

        print("Model save path:", final_model_save_path)
        eval_model = create_model(sess, args, constants.EVAL, vocab_size, load_pretrained_model=True,
                                  model_save_dir=final_model_save_path)
        infer_model = create_model(sess, args, constants.INFER, vocab_size, load_pretrained_model=True,
                                   model_save_dir=final_model_save_path, reuse=True)
        print("INFERENCE TYPE:%s" % args.decode_type)
        inference(infer_model, sess=sess, args=args, src_test_iterator=src_test_iterator,
                  decoder_s=constants.SENT_LIST, src_test_next=src_test_iterator.get_next(),
                  vocab_rev=vocab_rev, result_dir=args.final_tsf_result_dir)
