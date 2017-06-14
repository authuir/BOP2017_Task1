#! /usr/bin/env python3.5
# -*- coding: utf-8 -*-
# encoding='utf-8'

import numpy as np
import random
import sys

max_q_len  = 100
max_a_len  = 100
train_file = './data/BoP2017-DBQA.train.txt'
dev_file  = './data/BoP2017-DBQA.dev.txt'
test_file  = './data/BoP2017-DBQA.test.txt'

def build_vocab(train_file, dev_file):
    code = int(0)
    vocab = {}
    vocab[u'夨'] = code
    code += 1
    for line in open(train_file,'r', encoding='UTF-8'):
        items = line.strip().split('\t')
        for i in range(1, 3):
            for wrd in items[i]:
                if not wrd in vocab:
                    vocab[wrd] = code
                    code += 1
    for line in open(dev_file,'r', encoding='UTF-8'):
        items = line.strip().split('\t')
        for i in range(1, 3):
            for wrd in items[i]:
                if not wrd in vocab:
                    vocab[wrd] = code
                    code += 1
    for line in open(test_file,'r', encoding='UTF-8'):
        items = line.strip().split('\t')
        for i in range(0, 2):
            for wrd in items[i]:
                if not wrd in vocab:
                    vocab[wrd] = code
                    code += 1
    return vocab

def read_alist(file_name):
    alist = []
    for line in open(file_name,'r', encoding='UTF-8'):
        items = line.strip().split('\t')
        #print(items[2])
        alist.append(items[2])
    print('read_alist done ......')
    return alist

def read_raw(file_name):
    raw = []
    for line in open(file_name,'r', encoding='UTF-8'):
        items = line.strip().split('\t')
        if items[0] == '1':
            raw.append(items)
    return raw

def read_error_raw(file_name):
    raw = {}
    for line in open(file_name,'r', encoding='UTF-8'):
        items = line.strip().split('\t')
        if items[0] == '0':
            if not items[1] in raw:
                raw[items[1]] = []
            raw[items[1]].append(items[2])
    return raw

def rand_list(qalist):
    index = random.randint(0, len(qalist) - 1)
    return qalist[index]

def encode(vocab, string, size):
    x = []
    words = string
    for i in range(0, size):
        if (i < len(words)):
            if words[i] in vocab:
                x.append(vocab[words[i]])
            else:
                x.append(vocab[u'夨'])
                print("cannot recognize:",words[i])
        else:
            x.append(vocab[u'夨'])
    return x

def load_data_test(vocab, alist, raw, size):
    x_train_q = []
    x_train_ra = []
    x_train_wa = []
    for i in range(0, size):
        items = raw[random.randint(0, len(raw) - 1)]
        nega = rand_list(alist)
        x_train_q.append(encode(vocab, items[1], max_q_len))
        x_train_ra.append(encode(vocab, items[2], max_a_len))
        x_train_wa.append(encode(vocab, nega, max_a_len))
    return np.array(x_train_q), np.array(x_train_ra), np.array(x_train_wa)

def load_train_data(vocab, error_list, alist, raw, size):
    x_train_q = []
    x_train_ra = []
    x_train_wa = []
    for i in range(0, size):
        items = raw[random.randint(0, len(raw) - 1)]
        nega = rand_list(alist)
        if items[1] in error_list:
            nega = rand_list(error_list[items[1]])
        x_train_q.append(encode(vocab, items[1], max_q_len))
        x_train_ra.append(encode(vocab, items[2], max_a_len))
        x_train_wa.append(encode(vocab, nega, max_a_len))
    return np.array(x_train_q), np.array(x_train_ra), np.array(x_train_wa)

def load_data_by_qa(vocab, question, answer, batch):
    x_train_q = []
    x_train_ra = []
    x_train_wa = []
    for i in range(0, batch):
        x_train_q.append(encode(vocab, question, max_q_len))
        x_train_ra.append(encode(vocab, answer, max_a_len))
        x_train_wa.append(encode(vocab, answer, max_a_len))
    return np.array(x_train_q), np.array(x_train_ra), np.array(x_train_wa)

def reverse_vocab(vocab):
    x = {}
    for i in vocab:
        x[vocab[i]] = i
    return x

def test_util():
    vocab = build_vocab(train_file,test_file)
    alist = read_alist(train_file)
    raw = read_raw(train_file)
    errlist = read_error_raw(train_file)
    x_batch_1, x_batch_2, x_batch_3 = load_train_data(vocab, errlist, alist, raw, 50)
    #r_vocab = reverse_vocab(vocab)

if __name__ == '__main__':
    test_util()