# -*- coding: utf-8 -*-  
"""Standalone script to generate word vocabularies from monolingual corpus."""

from utils import constants
import tensorflow as tf


def build_vocab_from_file(src_file, save_path, min_frequency=5, size=0, without_sequence_tokens=False):
    """
    Generate word vocabularies from monolingual corpus.
    :param src_file: Source text file.
    :param save_path: Output vocabulary file.
    :param min_frequency: Minimum word frequency.  # for yelp and amazon, min_frequency=5
    :param size: Maximum vocabulary size. If = 0, do not limit vocabulary.
    :param without_sequence_tokens: If set, do not add special sequence tokens (start, end) in the vocabulary.
    :return: No return.
    """

    special_tokens = [constants.PADDING_TOKEN]
    if not without_sequence_tokens:
        special_tokens.append(constants.START_OF_SENTENCE_TOKEN)
        special_tokens.append(constants.END_OF_SENTENCE_TOKEN)

    vocab = {}
    with open(src_file) as f:
        for line in f:
            words = line.split('\t')[0].split(' ')
            for word in words:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

    filtered_list = filter(lambda kv: kv[1] > min_frequency, vocab.iteritems())
    sorted_list = sorted(filtered_list, key=lambda kv: (kv[1], kv[0]), reverse=True)
    if size != 0:
        sorted_list = sorted_list[:size]
    with open(save_path, 'w') as f:
        for s in special_tokens:
            f.write(s)
            f.write('\n')
        for (k, v) in sorted_list:
            f.write(k)
            f.write('\n')


def load_vocab(vocab_file):
    """Returns a lookup table and the vocabulary size."""

    def count_lines(filename):
        """Returns the number of lines of the file :obj:`filename`."""
        with open(filename, "rb") as f:
            i = 0
            for i, _ in enumerate(f):
                pass
            return i + 1

    vocab_size = count_lines(vocab_file) + 1  # Add UNK.
    vocab = tf.contrib.lookup.index_table_from_file(
        vocab_file,
        vocab_size=vocab_size - 1,
        num_oov_buckets=1)
    return vocab, vocab_size


def load_vocab_dict(vocab_file):
    """Returns a dictionary and the vocabulary size."""

    def count_lines(filename):
        """Returns the number of lines of the file :obj:`filename`."""
        with open(filename, "rb") as f:
            i = 0
            for i, _ in enumerate(f):
                pass
            return i + 1

    # vocab_size = count_lines(vocab_file) + 1  # Add UNK.

    vocab_dict = {}
    vocab_size = 0
    with open(vocab_file) as f:
        for line in f:
            word = line.strip()
            vocab_dict[word] = vocab_size
            vocab_size += 1
    vocab_dict[constants.UNKNOWN_TOKEN] = vocab_size
    vocab_size += 1
    return vocab_dict, vocab_size


def load_most_frquent_words(vocab_file, size):
    words = []
    with open(vocab_file) as f:
        for i, line in enumerate(f):
            if i >= size:
                break
            word = line.strip()
            words.append(word)
    return words

