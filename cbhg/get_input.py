#!/usr/bin/python3
# coding: utf-8

import os, sys
import numpy as np
import tensorflow as tf
# from .models import modules
from tensorflow.contrib.rnn import GRUCell

_pad = '_'

phone_set =[_pad] + [ '~', ';', '；', '、', ':', 'pau', '：', 'sp', '，', ',', 'sil', '？', '！', '。', '!', '?', 'n', 's', 'ER', 'iong', 'AE',
                      'HH', 'h', 'S', 'JH', 'AY', 'W', 'DH', 'SH', 't', 'AA', 'c', 'EY', 'j', 'ian', 'x', 'uan', 'ou', 'T', 'l', 'UH', 'D',
                      'e', 'sh', 'ang', 'ong', 'in', 'iao', 'ing', 'IH', 'z', 'van', 'uei', 'ei', 'AW', 'i', 'ch', 'OW', 'iang', 'eng', 'g',
                      've', 'K', 'M', 'P', 'ie', 'AH', 'Z', 'q', 'N', 'sil', 'AO', 'Y', 'f', 'uai', 'k', 'G', 'uo', 'F', 'ZH', 'OY', 'r',
                      'm', 'b', 'o', 'iou', 'zh', 'ao', 'EH', 'B', 'V', 'uang', 'er', 'CH', 'd', 'UW', 'en', 'AX', 'a', 'xr', 'iii', 'ua',
                      'TH', 'ueng', 'ia', 'NG', 'R', 'v', 'an', 'L', 'u', 'ai', 'ii', 'p', 'IY', 'uen', 'vn']

# Code-switch tone set
tone_set = [_pad] + ['0', '1', '2',
                     '3', '4', '5', '6', '7', '10', '11', '12']
# Word segmentation tags
seg_tag_set = [_pad] + ['B', 'M', 'E', 'S']
# Prosody set for phoneme and punctuation
prosody_set = [_pad] + ['0', '1', '2', '3', '4']

phone_set_d = {}
tone_set_d = {}
seg_tag_set_d = {}
prosody_set_d = {}

for i in range(len(phone_set)):
	# phone_set_d[phone_set[i]] = np.eye(len(phone_set))[i]
	phone_set_d[phone_set[i]] = i

for a in range(len(tone_set)):
	# tone_set_d[tone_set[a]] = np.eye(len(tone_set))[a]
    tone_set_d[tone_set[a]] = a

for b in range(len(seg_tag_set)):
	# seg_tag_set_d[seg_tag_set[b]] = np.eye(len(seg_tag_set))[b]
	seg_tag_set_d[seg_tag_set[b]] = b

for c in range(len(prosody_set)):
	# prosody_set_d[prosody_set[c]] = np.eye(len(prosody_set))[c]
	prosody_set_d[prosody_set[c]] = c
# print(phone_set_d)

#path_lab = './shuqi/labels'
#path_mel = './shuqi/mels'

#path_lab = '/home/wangpei/PycharmProjects/shuqi/labels'
#path_mel = '/home/wangpei/PycharmProjects/shuqi/mels

path_lab = '/home/team08/shuqi/labels'
path_mel = '/home/team08/shuqi/mels'

# print(path_lab)
dir_lab = os.listdir(path_lab)
dir_mel = os.listdir(path_mel)

inputs_lab = []
inputs_lab = np.array(inputs_lab)
inputs_mel = []
inputs_mel = np.array(inputs_mel)


total_len = 0
except_count = 0
for file in dir_lab[:5000]:
    file_name = os.path.splitext(file)[0]
    # print("file:", file_name)
    file_name_mel = os.path.join(path_mel, file_name + '.npy')
    path = os.path.join(path_lab, file)

    with open(path,encoding='utf-8') as f:
        lines = []
        try:
            for line in f.read().split('\n'):
                line = line.split('\t')
                # print(line[0])
                line[0] = phone_set_d[line[0]]
                line[1] = tone_set_d[line[1]]
                line[2] = seg_tag_set_d[line[2]]
                line[3] = prosody_set_d[line[3]]
                line[4] = float(line[4])
                line[5] = float(line[5])
                lines.append(line)
        except KeyError:
            except_count += 1
            if except_count % 100 == 0 :
                print("no kye : \" \" %s" % file)
            # if(line[0] != ' '):
        else:
            pass
        lines_len = len(lines)
    # print(lines)
    # print("ll", lines_len)
    array_mel = np.load(file_name_mel)
    mel_len = array_mel.shape[0]
    min_len = min(lines_len, mel_len)
    total_len += min_len-1
    lines = lines[:min_len-1]
    lines = np.array(lines)
    array_mel = array_mel[:min_len-1]
    inputs_lab  = np.append(inputs_lab, lines)
    inputs_mel = np.append(inputs_mel, array_mel)
    # print('mel_len',mel_len)
#inputs_lab = np.array(inputs_lab)
inputs_lab = inputs_lab.reshape(total_len, 6)
#print(inputs_lab.shape)
# inputs_lab[:,:,4:] = inputs_lab[:,:,4:].astype  ('float32')
inputs_mel = inputs_mel.reshape(total_len,80)

# input_phone_set_ids = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])

# print(np.array(a).shape)
# print(sess.run(input_seg_embedding, feed_dict={input_seg_ids:inputs_lab[:,:,2]}))
# print("phone_em:", sess.run(input_phone_set_embedding, feed_dict={input_phone_set_ids:inputs_lab[:,0],
#                                                input_tone_ids:inputs_lab[:1],
#                                                input_seg_ids:inputs_lab[:,2],
#                                                input_prsd_ids:inputs_lab[:,3]}))
# print("tone_em:", sess.run(input_tone_embedding, feed_dict={input_phone_set_ids:inputs_lab[:,0],
#                                                input_tone_ids:inputs_lab[:1],
#                                                input_seg_ids:inputs_lab[:,2],
#                                                input_prsd_ids:inputs_lab[:,3]}))
# print("prsd_em:", sess.run(input_prsd_embedding,feed_dict={input_phone_set_ids:inputs_lab[:,0],
#                                                input_tone_ids:inputs_lab[:1],
#                                                input_seg_ids:inputs_lab[:,2],
#                                                input_prsd_ids:inputs_lab[:,3]}))
# print("test", input_concat.shape)
# print("mel", inputs_mel.shape)

def cbhg(inputs, is_training, scope, K, projections, depth):
    #inputs.reshape(1,None,578)
    with tf.variable_scope(scope):
        with tf.variable_scope('conv_bank'):
            # Convolution bank: concatenate on the last axis to stack channels from all convolutions
            conv_outputs = tf.concat(
                [conv1d(inputs, k, 128, tf.nn.relu, is_training, 'conv1d_%d' % k) for k in range(1, K+1)],
                axis=-1
            )

        # Maxpooling:
        maxpool_output = tf.layers.max_pooling1d(
            conv_outputs,
            pool_size=2,
            strides=1,
            padding='same')

        # Two projection layers:
        proj1_output = conv1d(maxpool_output, 3, projections[0], tf.nn.relu, is_training, 'proj_1')
        proj2_output = conv1d(proj1_output, 3, projections[1], None, is_training, 'proj_2')

        # Residual connection:
        highway_input = proj2_output + inputs

        half_depth = depth // 2
        assert half_depth*2 == depth, 'encoder and postnet depths must be even.'

        # Handle dimensionality mismatch:
        if highway_input.shape[2] != half_depth:
            highway_input = tf.layers.dense(highway_input, half_depth)

        # 4-layer HighwayNet:
        for i in range(4):
            highway_input = highwaynet(highway_input, 'highway_%d' % (i+1), half_depth)
        rnn_input = highway_input

        # Bidirectional RNN
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            GRUCell(half_depth),
            GRUCell(half_depth),
            rnn_input,
            sequence_length=None,
            dtype=tf.float32)
        return tf.concat(outputs, axis=2)  # Concat forward and backward


def highwaynet(inputs, scope, depth):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        H = tf.layers.dense(
            inputs,
            units=depth,
            activation=tf.nn.relu,
            name='H')
        T = tf.layers.dense(
            inputs,
            units=depth,
            activation=tf.nn.sigmoid,
            name='T',
            bias_initializer=tf.constant_initializer(-1.0))
        return H * T + inputs * (1.0 - T)


def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
    with tf.variable_scope(scope):
        conv1d_output = tf.layers.conv1d(
            inputs,
            filters=channels,
            kernel_size=kernel_size,
            activation=activation,
            padding='same')
        return tf.layers.batch_normalization(conv1d_output, training=is_training)

def prenet_yhhc(input):
    out = tf.nn.relu(tf.layers.dense(input, 578))
    out = tf.layers.dense(out, 578)
    return out

def prenet2_yhhc(input):
    out = tf.nn.relu(tf.layers.dense(input, 256))
    out = tf.layers.dense(out, 80)
    return out

def cbhg_yhhc(input_lab):
    input_lab = prenet_yhhc(input_lab)
    return cbhg(inputs_lab,
                        True,
                        scope='cbhg_yhhc',
                        K=8,
                        projections=[256, 256],
                        depth=256)

batch_size = 25
seq_length = 150
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.97
n_batch = len(inputs_lab) // (seq_length*batch_size)


input_x = tf.compat.v1.placeholder(tf.float32, [None,seq_length,6])
mel_y = tf.compat.v1.placeholder(tf.float32, [None, seq_length, 80])

phone_set_embedding = tf.Variable(tf.random.truncated_normal([118, 448], stddev =0.1))
input_phone_set_embedding = tf.nn.embedding_lookup(phone_set_embedding, tf.to_int32(input_x[:,:,0]))

tone_embedding = tf.Variable(tf.random.truncated_normal([12, 64], stddev=0.1))
input_tone_embedding = tf.nn.embedding_lookup(tone_embedding, tf.to_int32(input_x[:,:,1]))

seg_embedding = tf.Variable(tf.random.truncated_normal([5, 32], stddev=0.1))
input_seg_embedding = tf.nn.embedding_lookup(seg_embedding, tf.to_int32(input_x[:,:,2]))

prsd_embedding = tf.Variable(tf.random.truncated_normal([6, 32], stddev=0.1))
input_prsd_embedding = tf.nn.embedding_lookup(prsd_embedding, tf.to_int32(input_x[:,:,3]))

e = tf.to_float(input_x[:,:,4:])

input_concat = tf.concat([input_phone_set_embedding,input_tone_embedding,input_seg_embedding,input_prsd_embedding,e],axis=-1)

cbhg_out = cbhg(input_concat,True,scope='cbhg_yhhc',K=8,projections=[256, 578],depth=256)
cbhg_out = prenet2_yhhc(cbhg_out)

loss_batch = tf.reduce_mean(tf.squared_difference(cbhg_out, mel_y))
global_step = tf.Variable(0, trainable=False)


learning_rate = tf.compat.v1.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                                     n_batch, LEARNING_RATE_DECAY)
train = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_batch, global_step=global_step)
# out_y = cbhg_yhhc()
tf.summary.scalar('loss', loss_batch)
merge_summary = tf.summary.merge_all()

saver = tf.train.Saver()
import time
	
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(100):
        for n in range(n_batch):
            batch_x = []
            batch_x = np.array(batch_x)
            batch_y = []
            batch_y = np.array(batch_y)
            for i in range(batch_size):
                x = inputs_lab[(n*batch_size+i)*seq_length:(n*batch_size+i+1)*seq_length,:]
                y = inputs_mel[(n*batch_size+i)*seq_length:(n*batch_size+i+1)*seq_length,:]
                batch_x = np.append(batch_x,x)
                batch_y = np.append(batch_y,y)
            batch_x = batch_x.reshape(batch_size, seq_length, 6)
            batch_y = batch_y.reshape(batch_size, seq_length, 80)
            # print(batch_x.shape)
            sess.run(train, feed_dict={input_x:batch_x,mel_y: batch_y})

            if  n % 10 == 0:
                print("smallloss %d :" %n, sess.run(loss_batch, feed_dict={input_x:batch_x, mel_y: batch_y}))
        saver.save(sess, "model.ckpt")
        train_writer = tf.summary.FileWriter('logs/', sess.graph)
    # npy_idx = 1
    # for file in dir_lab[1:6]:
    #     file_name = os.path.splitext(file)[0]
    #     # print("file:", file_name)
    #     file_name_mel = os.path.join(path_mel, file_name + '.npy')
    #     path = os.path.join(path_lab, file)
    #     with open(path,encoding='utf-8') as f:
    #         lines = []
    #         try:
    #             for line in f.read().split('\n'):
    #                 line = line.split('\t')
    #                 # print(line[0])
    #                 line[0] = phone_set_d[line[0]]
    #                 line[1] = tone_set_d[line[1]]
    #                 line[2] = seg_tag_set_d[line[2]]
    #                 line[3] = prosody_set_d[line[3]]
    #                 line[4] = float(line[4])
    #                 line[5] = float(line[5])
    #                 lines.append(line)
    #         except KeyError:
    #             print("no kye : \" \" %s" % file)
    #             # if(line[0] != ' '):
    #         else:
    #             pass
    #         lines_len = len(lines)
    #     lines = np.array(lines)
    #     print(lines.shape)
    #     input_phone_set_ids_test = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])
    #     phone_set_embedding_test = tf.Variable(tf.random.truncated_normal([118, 448], stddev =0.1))
    #     input_phone_set_embedding_test = tf.nn.embedding_lookup(phone_set_embedding, input_phone_set_ids_test)

    #     input_tone_ids_test = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])
    #     tone_embedding_test = tf.Variable(tf.random.truncated_normal([12, 64], stddev=0.1))
    #     input_tone_embedding_test = tf.nn.embedding_lookup(tone_embedding, input_tone_ids_test)

    #     input_seg_ids_test = tf.placeholder(dtype=tf.int32, shape=[None])
    #     seg_embedding_test = tf.Variable(tf.random.truncated_normal([5, 32], stddev=0.1))
    #     input_seg_embedding_test = tf.nn.embedding_lookup(seg_embedding, input_seg_ids_test)

    #     input_prsd_ids_test = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])
    #     prsd_embedding_test = tf.Variable(tf.random.truncated_normal([6, 32], stddev=0.1))
    #     input_prsd_embedding_test = tf.nn.embedding_lookup(prsd_embedding, input_prsd_ids_test)

    #     sess = tf.compat.v1.InteractiveSession()
    #     sess.run(tf.global_variables_initializer())
    #     # print(seg_embedding.eval())

    #     a_test = sess.run(input_phone_set_embedding_test, feed_dict={input_phone_set_ids_test:lines[:,0]})
    #     b_test = sess.run(input_tone_embedding_test, feed_dict={input_tone_ids_test:lines[:,1]})
    #     c_test = sess.run(input_seg_embedding_test, feed_dict={input_seg_ids_test:lines[:,2]})
    #     d_test = sess.run(input_prsd_embedding_test, feed_dict={input_prsd_ids_test:lines[:,3]})
    #     e_test = lines[:,4:].astype(np.float32)
    #     # f = lines[:,:,5].astype(np.float32)
    #     input_concat_test = sess.run(tf.concat([tf.to_float(a_test),b_test,c_test,d_test,e_test],axis=-1))
    #     print(input_concat_test.shape)
    #     test_result = sess.run(cbhg_out, feed_dict = {input_x: [input_concat_test[:seq_length]]})
    #     print(test_result.shape)
    #     path = 'test_mel/' + '00000' + str(npy_idx) + '.npy'
    #     np.save(path,test_result)
    #     npy_idx = npy_idx + 1


