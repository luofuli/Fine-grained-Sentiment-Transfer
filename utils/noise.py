# -*- coding: utf-8 -*-
import numpy as np
import random


def add_noise_(words, vocab=None, p=0.1, d=3):
    """Add noise to single data"""
    def _insert_noise(words, vocab, p, freq=None):
        if freq is None:
            freq = 100
        res = []
        for word in words:
            r = random.uniform(0, 1)
            if r < p:
                res += [random.sample(vocab[:freq], 1)[0], word]
            else:
                res += [word]
        return res

    def _delete_noise(words, p):
        res = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                res += [word]
        return res

    def _order_noise(words, d):
        words = np.array(words)
        cor = list(range(0, d + 1))
        new_idxs = np.argsort(np.array([i + random.sample(cor, 1)[0] for i in range(len(words))]))
        return list(words[new_idxs])

    if vocab is not None:
        words = _insert_noise(words, vocab, p)
    words = _delete_noise(words, p)
    words = _order_noise(words, d)
    return words


def add_noise(batch_ids, sequence_length, version=2):
    """Wraps add_noise_python for a batch of data."""
    for i, (ids, length) in enumerate(zip(batch_ids, sequence_length)):
        noisy_ids = add_noise_(ids[:length])
        noisy_sequence_length = len(noisy_ids)
        batch_ids[i][:noisy_sequence_length] = noisy_ids
        batch_ids[i][noisy_sequence_length:] = [0] * (len(ids) - noisy_sequence_length)
        sequence_length[i] = noisy_sequence_length
    return batch_ids, sequence_length


if __name__ == "__main__":
    pass
