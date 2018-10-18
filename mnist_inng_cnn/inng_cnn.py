#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 10:23:41 2018

@author: zhanghuangzhao
"""

import tensorflow as tf
import numpy
import random

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

def build_cnn_arch(lr=0.001, lr_neg=0.001, img_h=28, img_w=28, img_c=1,
                   padding_setting="SAME",
                   kernel_settings=[[4,4,1,4], [4,4,4,8], [4,4,8,16]],
                   stride_settings=[[1,2,2,1], [1,2,2,1], [1,2,2,1]],
                   fm_h=4, fm_w=4, fm_c=16, fc_width=[256, 256],
                   neg_candidate=None, neg_candidate_n=5000):
    
    x = tf.placeholder(tf.float32, shape=[None, img_h * img_w * img_c],
                       name="input")
    tf.add_to_collection("model", x)
    y = tf.placeholder(tf.int32, shape=[None], name="label_idx")
    tf.add_to_collection("model", y)
    x_reshape = tf.reshape(x, [-1, img_h, img_w, img_c],
                           name="input_reshape")
    
    kernels = []
    for i in range(len(kernel_settings)):
        kernels.append(tf.Variable(tf.random_normal(kernel_settings[i], 0, 0.1),
                                   name="kernel_"+str(i)))
        tf.add_to_collection("model", kernels[-1])
    
    tensors = [x_reshape]
    tf.add_to_collection("model", tensors[-1])
    for i in range(len(kernels)):
        tensors.append(tf.nn.conv2d(tensors[-1], kernels[i], stride_settings[i],
                                    padding=padding_setting, name="conv_"+str(i)))
        tf.add_to_collection("model", tensors[-1])
    
    fc_layers = []
    for n_l in fc_width:
        fc_layers.append(tf.layers.Dense(n_l, activation="relu",
                                         name="fc_"+str(len(fc_layers))))
        tf.add_to_collection("model", fc_layers[-1])
    fc_layers.append(tf.layers.Dense(2, activation=None,
                                     name="logit"))
    tf.add_to_collection("model", fc_layers[-1])
    
    tensors.append(tf.reshape(tensors[-1], [-1, fm_h*fm_w*fm_c],
                              name="conv_last_reshape"))
    tf.add_to_collection("model", tensors[-1])
    for l in fc_layers:
        tensors.append(l(tensors[-1]))
        tf.add_to_collection("model", tensors[-1])
    tensors.append(tf.nn.softmax(tensors[-1], axis=-1, name="prob"))
    tf.add_to_collection("model", tensors[-1])
    tensors.append(tf.argmax(tensors[-1], axis=-1,
                             output_type=tf.int32, name="output"))
    tf.add_to_collection("model", tensors[-1])
    
    y_onehot = tf.one_hot(y, 2, on_value=1, off_value=0, name="label_onehot")
    tf.add_to_collection("model", y_onehot)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_onehot,
                                                      logits=tensors[-3],
                                                      name="cross_entropy_loss")
    tf.add_to_collection("model", loss)
    opt = tf.train.GradientDescentOptimizer(lr, name="optimizer_gd")
    tf.add_to_collection("model", opt)
    var_list = tf.trainable_variables()
    train_op = opt.minimize(loss, var_list=var_list)
    tf.add_to_collection("model", train_op)
    
    accurate = tf.cast(tf.equal(y, tensors[-1], name="accurate"),
                       dtype=tf.float32, name="accurate_float")
    tf.add_to_collection("model", accurate)
    accuracy = tf.reduce_mean(accurate, name="accuracy")
    tf.add_to_collection("model", accuracy)
    
    arch = {"in": x, "label": y, "out": tensors[-1], "prob": tensors[-2],
            "loss": loss, "acc": accuracy, "op": train_op}
    
    
    if neg_candidate is None:
        neg_x = tf.Variable(tf.random_normal([neg_candidate_n, img_h * img_w * img_c],
                                             mean=0.5, stddev=0.1), 
                            dtype=tf.float32, name="neg_sample")
    else:
        neg_x = tf.Variable(neg_candidate, dtype=tf.float32, name="neg_sample")
    tf.add_to_collection("neg", neg_x)
    neg_y = tf.Variable(tf.ones([neg_candidate_n], dtype=tf.int32, name="neg_label"))
    tf.add_to_collection("neg", neg_y)
    neg_x_reshape = tf.reshape(neg_x, [-1, img_h, img_w, img_c],
                               name="neg_sample_reshape")
    
    neg_tensors = [neg_x_reshape]
    tf.add_to_collection("neg", neg_tensors[-1])
    for i in range(len(kernels)):
        neg_tensors.append(tf.nn.conv2d(neg_tensors[-1], kernels[i],
                                        stride_settings[i],
                                        padding=padding_setting,
                                        name="neg_conv_"+str(i)))
        tf.add_to_collection("neg", neg_tensors[-1])

    neg_tensors.append(tf.reshape(neg_tensors[-1], [-1, fm_h * fm_w * fm_c],
                                  name="neg_conv_last_reshape"))
    tf.add_to_collection("neg", neg_tensors[-1])
    for l in fc_layers:
        neg_tensors.append(l(neg_tensors[-1]))
        tf.add_to_collection("neg", neg_tensors[-1])
    neg_tensors.append(tf.nn.softmax(neg_tensors[-1], axis=-1, name="neg_prob"))
    tf.add_to_collection("neg", neg_tensors[-1])
    neg_tensors.append(tf.argmax(neg_tensors[-1], output_type=tf.int32,
                                 name="neg_output"))
    tf.add_to_collection("neg", neg_tensors[-1])
    
    neg_y_onehot = tf.one_hot(neg_y, 2, on_value=1, off_value=0,
                              name="neg_label_onehot")
    tf.add_to_collection("neg", neg_y_onehot)
    neg_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=neg_y_onehot,
                                                          logits=neg_tensors[-3],
                                                          name="neg_cross_entropy_loss")
    tf.add_to_collection("neg", neg_loss)
    neg_opt = tf.train.GradientDescentOptimizer(lr_neg, name="neg_optimizer_gd")
    tf.add_to_collection("neg", neg_opt)
    neg_var_list = [neg_x]
    neg_train_op = neg_opt.minimize(neg_loss, var_list=neg_var_list)
    tf.add_to_collection("neg", neg_train_op)
    
    neg = {"neg": neg_x, "label": neg_y, "op": neg_train_op}
    
    return (arch, neg)

if __name__ == "__main__":
    
    dataset = input_data.read_data_sets("../mnist/")  
    model, neg = build_cnn_arch()
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    neg_dataset = sess.run(neg["neg"])
    neg_dataset = random.sample(list(neg_dataset), len(neg_dataset))
    
    neg_images = []
    
    for epoch in range(1000):
    
        for iteration in range(501):
            b_pos, _ = dataset.train.next_batch(40)
            if len(neg_dataset) < 24:
                neg_dataset = sess.run(neg["neg"])
                neg_dataset = random.sample(list(neg_dataset), len(neg_dataset))
            b_neg = neg_dataset[:24]
            neg_dataset = neg_dataset[24:]
            b_x = numpy.concatenate((b_pos, b_neg), axis=0)
            b_y = numpy.asarray([1 if i < 40 else 0 for i in range(64)])
            sess.run(model["op"], feed_dict={model["in"]: b_x,
                                             model["label"]: b_y})
            if iteration % 100 == 0:
                print("\titer %d, acc = %.4f" %
                      (iteration, sess.run(model["acc"],
                                           feed_dict={model["in"]: b_x,
                                                      model["label"]: b_y})))
    
        for iteration in range(100):
            sess.run(neg["op"])
            print("\repoch %d %d" % (epoch, iteration+1), end="")
        print()
        
        b = sess.run(neg['neg'])
        neg_images.append(b[:10].reshape(10, 28, 28))
        b = numpy.reshape(b[:10], [10, 784])
        p = sess.run(model['out'], feed_dict={model["in"]: b})
        print("epoch %d, prediction is %s" % (epoch, str(p)))
        plt.imshow(neg_images[-1].transpose((1,0,2)).reshape((28, 10*28)))
        plt.imsave("demo_figure/"+str(epoch)+".jpg",
                   neg_images[-1].transpose((1,0,2)).reshape((28, 10*28)))
        plt.show()