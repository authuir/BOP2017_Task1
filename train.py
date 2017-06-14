#! /usr/bin/env python3.5
# -*- coding: utf-8 -*-
# encoding='utf-8'

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import util
from net import CNN
import operator

##########################################################################
#  参数
##########################################################################

# 模型参数
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 500, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")
# 训练参数
tf.flags.DEFINE_integer("batch_size", 30, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 25000, "Epochs (default: 200)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
# Session参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


##########################################################################
#  数据载入
##########################################################################

print("Loading data...")

# 语料库
vocab      = util.build_vocab(util.train_file, util.dev_file)
# 反向语料库
r_vocab    = util.reverse_vocab(vocab)
# 训练集的答案list
alist      = util.read_alist(util.train_file)
# 正确答案的问题和答案
raw        = util.read_raw(util.train_file)
# 正确答案的问题和答案
errlist  = util.read_error_raw(util.train_file)

x_train_q, x_train_ra, x_train_wa = util.load_data_test(vocab, alist, raw, FLAGS.batch_size)

print('x_train_q', np.shape(x_train_q))
print("Load done...")


##########################################################################
#  训练/测试
##########################################################################

with tf.Graph().as_default():
  with tf.device("/gpu:0"):
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = CNN(
            sequence_length=x_train_q.shape[1],
            batch_size=FLAGS.batch_size,
            vocab_size=len(vocab),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # 优化器
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-1)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # 保存配置
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # 初始化变量
        sess.run(tf.global_variables_initializer())

        # Restore The old version

        # Model 0
        #saver.restore(sess, "./runs/1496213562/checkpoints/model-25000")

        # Model 01
        #saver.restore(sess, "./runs/1496219924/checkpoints/model-25000")

        # Model 02
        #saver.restore(sess, "./runs/1496238525/checkpoints/model-25000")

        # Model 02A
        #saver.restore(sess, "./runs/1496242239/checkpoints/model-25000")

        # Model 03
        #saver.restore(sess, "./runs/1496306200/checkpoints/model-20000")

        # Model Using
        saver.restore(sess, "./runs/1496636830/checkpoints/model-25000")

        def train_step(input_question, input_r_answer, input_w_answer):
            """
            训练步骤
            """
            feed_dict = {
              cnn.input_question: input_question,
              cnn.input_r_answer: input_r_answer,
              cnn.input_w_answer: input_w_answer,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        def test(file_to_test):
            """
            训练 
            """
            def get_score_by_qa(question, answer):
                x_test_1, x_test_2, x_test_3 = util.load_data_by_qa(vocab, question, answer, FLAGS.batch_size)
                feed_dict = {
                    cnn.input_question: x_test_1,
                    cnn.input_r_answer: x_test_2,
                    cnn.input_w_answer: x_test_3,
                    cnn.dropout_keep_prob: 1.0
                }
                batch_score = sess.run([cnn.cos_12], feed_dict)
                return batch_score[0][0]
            def dict2list(dic:dict):
                keys = dic.keys()
                vals = dic.values()
                lst = [(key, val) for key, val in zip(keys, vals)]
                return lst
            def sortedDictValues(adict): 
                return sorted(dict2list(adict), key=lambda x:x[0], reverse=True)

            def mmr_calc(final_score, final_score_count, answer_score):
                answer_score = sortedDictValues(answer_score)
                cnt = int(1)
                rank = int(0)
                for item in answer_score:
                    print("{}: {} : {}".format(cnt,item[0],item[1]))
                    if (right_answer == item[1]):
                        rank = cnt
                    cnt = cnt + 1
                if rank >= 1:
                    final_score = final_score + 1.0/float(rank)
                    final_score_count = final_score_count + 1
                    MRR = final_score/float(final_score_count)
                    if cnt != 1:
                        print(u"正确结果排名/总数 ：{}/{}".format(rank,cnt-1))
                        print(u"Question count ： {}".format(final_score_count))
                        print(u"MRR ： {}\n".format(MRR))
                return final_score, final_score_count

            temp_question_vector = u"null"
            right_answer = u"null"
            answer_score = {}
            final_score = 0.0
            final_score_count = 0
            #outfile = open("./out/outfile",'w', encoding='UTF-8'):
            for line in open(file_to_test,'r', encoding='UTF-8'):
                items = line.strip().split('\t')
                question = items[1]
                answer   = items[2]
                score    = get_score_by_qa(question, answer)
                #outfile.write(str(score))
                if question != temp_question_vector:
                    print("Question:{}".format(temp_question_vector))
                    final_score, final_score_count = mmr_calc(final_score, final_score_count, answer_score)

                    answer_score.clear()
                    answer_score = {}
                    temp_question_vector = question
                    answer_score[score] = answer
                else:
                    answer_score[score] = answer

                if items[0] == '1':
                    right_answer = answer
                else:
                    pass
            #outfile.close()

            print("Question:{}".format(temp_question_vector))
            final_score, final_score_count = mmr_calc(final_score, final_score_count, answer_score)

            MRR = final_score/float(final_score_count)
            print("Final MRR:{}".format(MRR))
            print("Num of QA:{}".format(final_score_count))

        def train():
            """
            训练 
            """
            for i in range(FLAGS.num_epochs):
                try:
                    x_batch_q, x_batch_ra, x_batch_wa = util.load_train_data(vocab, errlist, alist, raw, FLAGS.batch_size)
                    train_step(x_batch_q, x_batch_ra, x_batch_wa)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                except Exception as e:
                    print(e)

        #train()
        test('./data/BoP2017-DBQA.dev.txt')
