#!/usr/bin/python
# coding: utf-8
import os
import numpy as np
import tensorflow as tf

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
	phone_set_d[phone_set[i]] = np.eye(len(phone_set))[i]

for a in range(len(tone_set)):
	tone_set_d[tone_set[a]] = np.eye(len(tone_set))[a]

for b in range(len(seg_tag_set)):
	# seg_tag_set_d[seg_tag_set[b]] = np.eye(len(seg_tag_set))[b]
	seg_tag_set_d[seg_tag_set[b]] = b

for c in range(len(prosody_set)):
	prosody_set_d[prosody_set[c]] = np.eye(len(prosody_set))[c]



print(phone_set_d)
