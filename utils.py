
# -*- coding: utf-8 -*-
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_sents(sents, pad_token):

    max_len = len(max(sents, key=lambda x: len(x)))

    return [pad_sent(sent, pad_token, max_len) for sent in sents]


def pad_sent(sent, pad_token, max_len):
    return [sent[i] if i < len(sent) else pad_token for i in range(max_len)]


def test_pad_sents():
    sents = [['a', 'b', 'c'], ['a', 'b', 'c', 'd', 'e'], ['a'], ['a', 'b'], ['a', 'b', 'c', 'd']]
    sents_padded = pad_sents(sents, pad_token='x')
    assert tuple(sents_padded[0]) == ('a', 'b', 'c', 'x', 'x'), 'something wrong ...'


def read_corpus(file_path, source):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents


if __name__ == '__main__':
    test_pad_sents()

