# -*- coding: utf-8 -*-
"""
 @version: python2.7
 @author: luofuli
 @time: 2018/8/4 21:54
"""
# -*- coding: utf-8 -*-
import pickle
import time
import tensorflow as tf
import sys, os
import numpy as np

sys.path.append('..')
from model import Regressor
from utils import constants
from options import *
from utils.data import load_dataset
from utils.vocab import build_vocab_from_file, load_vocab

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_model(sess, args, vocab_size, mode=constants.TRAIN, reuse=None, load_pretrained_model=False):
    print("-- Create sentiment regressor (mode: %s) ---" % mode)

    with tf.variable_scope(constants.REG_VAR_SCOPE, reuse=reuse):
        model = Regressor(mode, args.__dict__, vocab_size)
    sess.run(tf.tables_initializer())

    if load_pretrained_model:
        reg_model_save_dir = args.reg_model_save_dir
        try:
            model.saver.restore(sess, reg_model_save_dir)
            print("Loading regression model from", reg_model_save_dir)
        except Exception as e:
            print("Error! Loading regression model from", reg_model_save_dir)
            model.saver.restore(sess, tf.train.latest_checkpoint(reg_model_save_dir))
            print("Loading model from", tf.train.latest_checkpoint(reg_model_save_dir))
    else:
        if reuse is None:
            print("Creating regression model with fresh parameters.")
            sess.run(tf.global_variables_initializer())
        else:
            print('Reuse parameters.')
    return model


def train():
    best = {
        "eval_loss": 100.0,
        "step": 0,
    }

    global_step = 0
    for i in range(args.n_epoch):
        sess.run([train_data_iterator.initializer])
        n_batch = 0
        t0 = time.time()
        while True:
            try:
                data = sess.run(train_data_next_op)
                feed_dict = {
                    train_model.x: data['ids'],
                    train_model.y: data['senti'],
                    train_model.sequence_length: data['length']}

                ops = [train_model.loss,
                       train_model.mae_loss,
                       train_model.predict_score,
                       train_model.y_true,
                       train_model.train_op]
                res = sess.run(ops, feed_dict=feed_dict)

                n_batch += 1
                global_step += 1
                if n_batch % 100 == 0:
                    eval_loss = eval()

                    if eval_loss < best["eval_loss"]:
                        best["eval_loss"] = eval_loss
                        best["step"] = global_step
                        print("Save model at: %s " % (args.reg_model_save_dir + str(global_step)))
                        train_model.saver.save(sess, args.reg_model_save_dir, global_step=global_step)

                    print("Epoch/n_batch/global_step:{}/{}/{}\tTrain_loss:{}\tEval_loss:{}\tTime:{}".
                          format(i, n_batch, global_step, res[0], eval_loss, time.time() - t0))
                    print("y_pred: min:%.3f max:%.3f mean:%.3f\ty_true: min:%.3f max:%.3f mean:%.3f"
                          % (np.min(res[2]), np.max(res[2]), np.mean(res[2]),
                             np.min(res[3]), np.max(res[3]), np.mean(res[3])))

                    if global_step - best["step"] > 5000:
                        print("--- Early stop! ---")
                        print("Best eval_loss: {}\n Best step: {}".format(best["eval_loss"], best["step"]))
                        return

            except tf.errors.OutOfRangeError as e:  # next epoch
                # print(e)
                print("Train---Total N batch:{}".format(n_batch))
                break


def eval():
    sess.run([dev_data_iterator.initializer])
    n_batch = 0
    t0 = time.time()
    losses = []
    while True:
        try:
            data = sess.run(dev_data_next_op)  # get real data!!
            feed_dict = {
                eval_model.x: data['ids'],
                eval_model.y: data['senti'],
                eval_model.sequence_length: data['length']}

            ops = [eval_model.loss]
            res = sess.run(ops, feed_dict=feed_dict)
            losses.append(res[0])
            n_batch += 1
        except tf.errors.OutOfRangeError as e:  # next epoch
            print("Eval---Total N batch:{}\tCost time:{}".format(n_batch, time.time() - t0))
            break
    return sum(losses) / len(losses)


def evaluate_file(sess, args, eval_model, vocab, files, batch_size=100, print_logs=False):
    eval_iterator = load_dataset(files, vocab, constants.EVAL, batch_size=batch_size,
                                 min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len, has_source=True)
    eval_next_op = eval_iterator.get_next()

    sess.run([eval_iterator.initializer])
    n_batch = 0
    t0 = time.time()
    losses = []
    while True:
        try:
            data = sess.run(eval_next_op)  # get real data!!
            feed_dict = {
                eval_model.x: data['ids'],
                eval_model.y: data['senti'],
                eval_model.sequence_length: data['length']}

            ops = [eval_model.loss]
            res = sess.run(ops, feed_dict=feed_dict)
            losses.append(res[0])
            n_batch += 1
        except tf.errors.OutOfRangeError as e:  # next epoch
            if print_logs:
                print("Test---Total N batch:{}\tCost time:{}".format(n_batch, time.time() - t0))
            break
    del eval_iterator
    del eval_next_op

    return np.mean(losses)


if __name__ == "__main__":
    args = load_reg_arguments()

    # Step 1: build vocab and load data
    if not os.path.isfile(args.vocab_file):
        build_vocab_from_file(args.train_data, args.vocab_file)
    vocab, vocab_size = load_vocab(args.vocab_file)
    print('Vocabulary size:%s' % vocab_size)

    vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(
        args.vocab_file,  # target vocabulary file(each lines has a word)
        vocab_size=vocab_size - constants.NUM_OOV_BUCKETS,
        default_value=constants.UNKNOWN_TOKEN)

    with tf.device("/cpu:0"):  # Input pipeline should always be place on the CPU.

        if args.mode == constants.TRAIN:
            train_data_iterator = load_dataset(args.train_data, vocab, constants.TRAIN, batch_size=args.batch_size,
                                               min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len)
            train_data_next_op = train_data_iterator.get_next()

            dev_data_iterator = load_dataset(args.dev_data, vocab, constants.EVAL, batch_size=100,
                                             min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len)
            dev_data_next_op = dev_data_iterator.get_next()

        test_data_iterator = load_dataset(args.test_data, vocab, constants.TEST, batch_size=100,
                                          min_seq_len=args.min_seq_len, max_seq_len=args.max_seq_len)
        test_data_next_op = test_data_iterator.get_next()

    # create session
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=tf_config)  # limit gpu memory; don't pre-allocate memory; allocate as-needed

    # Initial and build model
    if args.mode == constants.TRAIN:
        # prepare for saving trained model
        args.reg_model_save_dir = args.reg_model_save_dir
        print("Model save dir: " + args.reg_model_save_dir)
        if not os.path.exists(args.reg_model_save_dir):
            print('Creat model save dir: ' + args.reg_model_save_dir)
            os.makedirs(args.reg_model_save_dir)
        dump_args_to_yaml(args, args.reg_model_save_dir)

        train_model = create_model(sess, args, vocab_size, mode=constants.TRAIN, reuse=None)
        eval_model = create_model(sess, args, vocab_size, mode=constants.EVAL, reuse=True)

        # train model
        train()

    elif args.mode == constants.EVAL:
        # eval model
        eval_model = create_model(sess, args, vocab_size, mode=constants.EVAL, load_pretrained_model=True)
        loss = eval()
        print("Eval loss: %.3f" % loss)
