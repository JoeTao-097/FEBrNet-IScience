#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:35:54 2021

@author: joe
"""
import sklearn.svm
from base_tool import *
import os
import keras
import numpy as np
from keras.applications.mobilenet import preprocess_input
from PIL import Image
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.5
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


'''
feature score reduce method
'''


def softmax(x):
    """ softmax function """
    # x -= np.max(x, axis=1, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x))
    return x


def get_feature_map(img_batch, test_model, get_index=1):
    #    pred_class = test_model.predict(preprocess_input(img_batch), verbose=0)
    gap_layer = test_model.get_layer(index=get_index)
    # iterate = K.function([test_model.get_layer('xception').get_layer(index=0).input], [gap_layer.output])
    iterate = K.function([test_model.input], [gap_layer.output])

    #    intermediate_layer_model = Model(inputs=test_model.get_layer('densenet121').get_layer(index=0).input, outputs=last_conv_layer.output)
    #    intermediate_output = intermediate_layer_model.predict(preprocess_input(img))

    gap_layer_output_value = iterate([preprocess_input(img_batch)])[0]

    return gap_layer_output_value


def feature_with_entropy_reduce(weight, fmap):
    Wd = np.maximum(weight, 0)
    fmap_maxpool = np.max(fmap, axis=0)
    fmap_minpool = np.min(fmap, axis=0)
    n_frame = fmap.shape[0]
    n_channel = fmap.shape[-1]
    fmap = fmap.reshape((n_frame, 1, n_channel))
    fmap_maxpool = fmap_maxpool.reshape((1, n_channel))
    Sf_id = []
    Hx = 1e100
    Hx0 = 1e101
    while (Hx < Hx0):
        Hx0 = Hx
        if len(Sf_id) == 0:
            fe_selected = np.zeros(fmap.shape)
        else:
            fe_selected = np.max(fmap[Sf_id], 0).reshape((1, 1, n_channel))
            fe_selected = np.repeat(fe_selected, repeats=n_frame, axis=0)
        fe_i = np.concatenate((fmap, fe_selected), axis=1)
        fe_i = np.max(fe_i, 1)
        D = np.sum(Wd * (fmap_maxpool - fe_i), 1)
        Sf_id.append(np.argmin(D))
        fe_m = np.max(fmap[Sf_id], 0)
        p = softmax(np.array([np.sum(np.maximum(-weight, 0) * fe_m), np.sum(Wd * fe_m)]))[1]
        Hx = -p * np.log2(p)
        print(Hx0)
    print('end')
    return Sf_id[:-1], Hx0


def generate_feature_entropy_map(weight, fmap):
    P_id, Hxp = feature_with_entropy_reduce(weight, fmap)
    N_id, Hxn = feature_with_entropy_reduce(-weight, fmap)
    fmap_p = np.max(fmap[P_id], axis=0)
    fmap_n = np.max(fmap[N_id], axis=0)
    fmap_pn = np.vstack((fmap_p, fmap_n))
    # fmap_pn = np.concatenate((W*fmap_p, W*fmap_n), axis=0)
    fmap_pn = np.max(fmap_pn, axis=0)
    return fmap_pn, [P_id, N_id]


def generate_Rframes(pretrained_model, file, save_dir):
    batch_size = 64
    REZ = 224
    fc_weight = pretrained_model.get_layer(index=-1).get_weights()
    W1 = fc_weight[0][:, 1]
    W0 = fc_weight[0][:, 0]
    W = W1 - W0
    create_dir(save_dir)
    vid = read_avi(file)
    vid_224 = np.zeros((vid.shape[0], REZ, REZ, 3), dtype=np.uint8)
    for i in range(len(vid)):
        vid_224[i] = zero_pad(vid[i], REZ)
    n_frames = vid_224.shape[0]
    n_batch = int(n_frames / batch_size) + 1
    start = 0
    fmap = np.zeros((1, 1024))
    for i in range(n_batch):
        img = vid_224[start:min(vid_224.shape[0], start + batch_size)]
        fmap = np.vstack((fmap, get_feature_map(img, model, get_index=-2)))
        start += batch_size
    fmap = fmap[1:]
    fmap_pn, Xid = generate_feature_entropy_map(W, fmap)
    for id in Xid[0]:
        img = vid[id]
        img = Image.fromarray(img)
        img.save(save_dir + "No.{}_frame_{}.png".format(i, id))


if __name__ == '__main__':
    model = keras.models.load_model('./CNN Backbone weights/MobileNet/mobilenet_224.h5',
                                    custom_objects={'focal_loss': focal_loss})
    model.summary()
    file = './DemoVideos/2.avi'
    save_dir = file.replace(file.split('.')[-1], '') + '\'s responsible frames/'
    generate_Rframes(model, file=file, save_dir=save_dir)
