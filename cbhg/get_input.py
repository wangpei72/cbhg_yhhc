#!/usr/bin/python3
# coding: utf-8

import os, sys
import numpy as np
import tensorflow as tf
from tacotron.models import modules

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



path_lab = '/home/wangpei/PycharmProjects/shuqi/labels'
path_mel = '/home/wangpei/PycharmProjects/shuqi/mels'

# print(path_lab)
dir_lab = os.listdir(path_lab)
dir_mel = os.listdir(path_mel)

inputs_lab = []
inputs_mel = []

# print(dir_lab)
print(len(phone_set))
print(len(tone_set))
print(len(seg_tag_set))
print(len(prosody_set))

for file in dir_lab[:1]:
	file_name = os.path.splitext(file)[0]
	print("file:", file_name)
	file_name_mel = os.path.join(path_mel, file_name + '.npy')
	with open(os.path.join(path_lab, file)) as f:
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
			print("no kye : \" \"")
			# if(line[0] != ' '):
		else:
			pass
		lines_len = len(lines)
	# print(lines)
	# print("ll", lines_len)
	array_mel = np.load(file_name_mel)
	mel_len = array_mel.shape[0]
	min_len = min(lines_len, mel_len)
	lines = lines[:min_len-1]
	array_mel = array_mel[:min_len-1]
	inputs_lab.append(lines)
	inputs_mel.append(array_mel)
	# print('mel_len',mel_len)
inputs_lab = np.array(inputs_lab)
# inputs_lab[:,:,4:] = inputs_lab[:,:,4:].astype('float32')
inputs_mel = np.array(inputs_mel)

input_phone_set_ids = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None])
phone_set_embedding = tf.Variable(tf.random.truncated_normal([118, 448], stddev =0.1))
input_phone_set_embedding = tf.nn.embedding_lookup(phone_set_embedding, input_phone_set_ids)

input_tone_ids = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None])
tone_embedding = tf.Variable(tf.random.truncated_normal([12, 64], stddev=0.1))
input_tone_embedding = tf.nn.embedding_lookup(tone_embedding, input_tone_ids)

input_seg_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])
seg_embedding = tf.Variable(tf.random.truncated_normal([5, 32], stddev=0.1))
input_seg_embedding = tf.nn.embedding_lookup(seg_embedding, input_seg_ids)

input_prsd_ids = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None,None])
prsd_embedding = tf.Variable(tf.random.truncated_normal([6, 32], stddev=0.1))
input_prsd_embedding = tf.nn.embedding_lookup(prsd_embedding, input_prsd_ids)

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.global_variables_initializer())
# print(seg_embedding.eval())
# print(inputs_lab.shape)
# print(inputs_lab[:,:,2].shape)
a = sess.run(input_phone_set_embedding, feed_dict={input_phone_set_ids:inputs_lab[:,:,0]})
b = sess.run(input_tone_embedding, feed_dict={input_tone_ids:inputs_lab[:,:,1]})
c = sess.run(input_seg_embedding, feed_dict={input_seg_ids:inputs_lab[:,:,2]})
d = sess.run(input_prsd_embedding, feed_dict={input_prsd_ids:inputs_lab[:,:,3]})
e = inputs_lab[:,:,4:].astype(np.float32)
# f = inputs_lab[:,:,5].astype(np.float32)
print(sess.run(tf.concat([a,b,c,d,e],axis=-1)))
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
# print("test", inputs_lab[0])
# print("mel", inputs_mel[0])

