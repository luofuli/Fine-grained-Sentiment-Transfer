# -*- coding: utf-8 -*-
import argparse
import sys
import os, re
from seq2sentiseq.options import *
from common_options import *


def add_cycle_arguments(parser):
    parser.add_argument('--n_epoch', default=30, type=int, help='Max n epoch during dual training.')
    parser.add_argument('--MLE_learning_rate', default=0.0001, type=float, help='Learning rate of MLE')
    parser.add_argument('--RL_learning_rate', default=0.000001, type=float, help='Learning rate of RL')

    # todo: 128, 64 will OOM
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size of dual training.')

    # for reward calculation
    parser.add_argument("--use_baseline", default=False, type=ast.literal_eval,
                        help="Use baseline in reward (input should be either 'True' or 'False'.)")
    parser.add_argument("--scale_cont_reward", default=True, type=ast.literal_eval,
                        help="Scale content score from 0 to 1 (input should be either 'True' or 'False'.)")
    parser.add_argument("--scale_senti_reward", default=True, type=ast.literal_eval,
                        help="Scale sentiment score from 0 to 1 (input should be either 'True' or 'False'.)")

    # for dual training
    parser.add_argument('--teacher_forcing', nargs='+', default=['back_trans_noise'],
                        help='Corpus used in teacher forcing (MLE), must in [`back_trans`, `back_trans_random`, '
                             '`back_trans_noise`, `pseudo_data`, `same`, `same_noise`]')
    parser.add_argument("--no_pretrain", action='store_true', help='No Pre-training for seq2seq model.')
    parser.add_argument('--no_RL', action='store_true', help='Without RL training.')

    # old pseudo decay arg
    parser.add_argument('--MLE_decay', action='store_true', help='Decay the use of pseudo data')  # Default is false
    parser.add_argument('--MLE_decay_type', choices=['linear', 'exp'], default='exp', help='Decay type (linear or exp).')
    parser.add_argument('--MLE_decay_rate', default=0.99, type=float, help='The decay rate.')
    parser.add_argument('--MLE_decay_steps', default=100, type=int, help='Decay every this many steps.')
    parser.add_argument('--MLE_initial_gap', default=1, type=int, help='Initial value.')
    
    parser.add_argument('--increase_beta', action='store_true', help='Increase the rate of ACC')
    parser.add_argument('--increase_step', default=1000, type=int, help='The increase step')

    # same decay rate should be larger
    parser.add_argument('--same_decay', action='store_true', help='Decay the use of pseudo data')
    parser.add_argument('--same_decay_type', choices=['linear', 'exp'], default='linear', help='Decay type (linear or exp).')
    parser.add_argument('--same_decay_rate', default=0.9, type=float, help='The decay rate.')
    parser.add_argument('--same_decay_steps', default=100, type=int, help='Decay every this many steps.')
    parser.add_argument('--same_initial_gap', default=2, type=int, help='Initial value.')

    # for logs and results
    parser.add_argument('--save_each_step', action='store_true')
    parser.add_argument('--save_per_step', default=1000, type=int, help='Save model per n steps')
    parser.add_argument('--eval_step', default=1000, type=int, help='Evaluate model.')  # todo: Enlarge this can train faster


def load_cycle_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_cycle_arguments(parser)
    add_common_arguments(parser)
    parser = parser.parse_args()
    return parser
