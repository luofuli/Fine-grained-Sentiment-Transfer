# -*- coding: utf-8 -*-  
"""  
 @version: python2.7 
 @author: luofuli 
 @time: 2018/8/6 21:11 
"""
import argparse
import sys
import time
import os, re
import ast
from yaml import load, dump
base_path = os.getcwd()  # current working directory
base_path_ = base_path.split('/')
base_path = '/'.join(base_path_[:base_path_.index('Fine-grained-Sentiment-Transfer') + 1])


dataset = "yelp"
pseudo = 'JS'


def add_common_arguments(parser):
    # data path
    parser.add_argument("--train_data", default="{}/data/{}/train.txt".format(base_path, dataset),
                        help="Two train files (absolute path).")
    parser.add_argument("--dev_data", default="{}/data/{}/dev.txt".format(base_path, dataset),
                        help="Two dev files (absolute path).")
    parser.add_argument("--test_data", default="{}/data/{}/test.txt".format(base_path, dataset),
                        help="Two test files (absolute path).")
    parser.add_argument("--pseudo_data", default="{}/data/{}/pseudo_{}_paired.txt".format(base_path, dataset, pseudo),
                        help="Pseudo parallel train files (absolute path).")
    parser.add_argument("--reference", default="{}/data/{}/reference.txt".format(base_path, dataset),
                        help="Reference file (absolute path).")

    # hyper-parameters of model
    parser.add_argument("--vocab_file", default="{}/data/{}/vocab".format(base_path, dataset), help="Vocabulary file.")
    parser.add_argument("--min_seq_len", default=3, help="Min sequence length.")
    parser.add_argument("--max_seq_len", default=30, help="Max sequence length.")
    parser.add_argument("--emb_dim", default=300, help="The dimension of word embeddings.")
    parser.add_argument("--semantic_embedding_size", default=300, help='Size of semantic embedding, same to --emb_dim')
    parser.add_argument("--sentiment_embedding_size", default=300, help='Size of sentiment embedding')
    parser.add_argument("--scale_sentiment", default=True, type=ast.literal_eval,
                        help="Scale sentiment score from 0 to 1 (input should be either 'True' or 'False'.)")

    # model save path
    parser.add_argument("--s2ss_model_save_dir", default="{}/tmp/model/{}/seq2sentiseq/".format(base_path, dataset),
                        help="Seq2sentiSeq model save dir.")
    parser.add_argument("--lm_model_save_dir", default='{}/tmp/model/{}/language_model/'.format(base_path, dataset),
                        help='Language model save dir.')
    parser.add_argument("--reg_model_save_dir", default='{}/tmp/model/{}/regressor/'.format(base_path, dataset),
                        help='Regression model save dir.')
    t0 = int(time.time()) % 1000
    parser.add_argument("--final_model_save_dir",
                        default="{}/tmp/model/{}/nmt_final-{}/".format(base_path, dataset, t0),
                        help="Final transfer model save dir")
    parser.add_argument("--tsf_result_dir", default="{}/tmp/output/{}".format(base_path, dataset),
                        help="Transfer result dir (before dual training).")
    parser.add_argument("--final_tsf_result_dir", default="{}/tmp/output/{}_final-{}".format(base_path, dataset, t0),
                        help="Final Transfer result dir (after dual training).")


def load_common_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_arguments(parser)
    return parser.parse_args()


def load_args_from_yaml(dir):
    args = load(open(os.path.join(dir, 'conf.yaml')))
    return args


def dump_args_to_yaml(args, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    dump(args, open(os.path.join(dir, 'conf.yaml'), 'w'))

