#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:37:38 2018

@author: zhanghuangzhao
"""

import tensorflow as tf
import numpy

import simple
import matplotlib.pyplot as plt

def build_arch(w_dim=2, w_hidden_layers=[10, 10, 10], cl_threshold=0.5, cl_lr=0.01, sy_lr=0.01,
               sy_size=2000, sy_mean=0., sy_stddev=5.,
               cl_scope="classification", sy_scope="synthesis"):
    
    layers = []
    for n_l in w_hidden_layers:
        layers.append(tf.layers.Dense(n_l, activation="relu",
                                      name="hidden_"+str(len(layers))))
    layers.append(tf.layers.Dense(1, activation=None,
                                  name="logit"))
        
    with tf.variable_scope(cl_scope, reuse=tf.AUTO_REUSE) as scope:
        
        x = tf.placeholder(tf.float32, shape=[None, w_dim], name="input")
        y = tf.placeholder(tf.float32, shape=[None, 1], name="label")
        states = [x]
        for l in layers:
            states.append(l(states[-1]))
        cl_prob = tf.nn.sigmoid(states[-1], name="output_prob")
        cl_out = tf.cast((cl_prob > cl_threshold), dtype=tf.float32, name="output_label")
            
        cl_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,
                                                          logits=states[-1],
                                                          name="loss")
        cl_trainable = [i for i in tf.trainable_variables()]
        cl_opt = tf.train.GradientDescentOptimizer(cl_lr, name="GDOpt")
        cl_train_op = cl_opt.minimize(cl_loss, var_list=cl_trainable)
        
        accurate = tf.equal(y, cl_out, name="accurate")
        accuracy = tf.reduce_mean(tf.cast(accurate, tf.float32), name="accuracy")
        
    with tf.variable_scope(sy_scope, reuse=tf.AUTO_REUSE) as scope:
        neg = tf.Variable(tf.random_normal(shape=[sy_size, w_dim],
                                           mean=sy_mean,
                                           stddev=sy_stddev,
                                           name="pseudo_neg"))
        sy_states = [neg]
        for l in layers:
            sy_states.append(l(sy_states[-1]))
            
        sy_loss = - tf.reduce_mean(sy_states[-1], name="loss")
        sy_trainable = [neg]
        sy_opt = tf.train.GradientDescentOptimizer(sy_lr, name="GDOpt")
        sy_train_op = sy_opt.minimize(sy_loss, var_list=sy_trainable)
    
    return {"cl_in": x, "cl_label": y, "cl_logit": states[-1],
            "cl_prob": cl_prob, "cl_out": cl_out, "cl_op": cl_train_op,
            "cl_loss": cl_loss, "cl_acc": accuracy,
            "sy_sample": neg, "sy_op": sy_train_op, 
            "sy_logit": sy_states[-1], "sy_loss": sy_loss}

if __name__ == "__main__":
    
    m = build_arch()
    d = simple.Simple()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    neg_data_seq = [sess.run(m["sy_sample"])]
    
    for epoch in range(300):
        
        d.set_neg(neg_data_seq[-1])
        plt.plot(neg_data_seq[-1][:,0], neg_data_seq[-1][:,1], ".")
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        plt.savefig("figure/"+str(epoch)+".jpg")
        plt.show()
        
        for i in range(1000):
            b = d.minibatch(32)
            sess.run(m["cl_op"], feed_dict={m["cl_in"]: b[0],
                                            m["cl_label"]: numpy.reshape(b[1], [32, 1])})
        b = d.minibatch(d.get_size())
        print("Epoch %d, acc = %.4f" % (epoch,
                                        sess.run(m["cl_acc"], feed_dict={m["cl_in"]: b[0],
                                                 m["cl_label"]: numpy.reshape(b[1], [d.get_size(), 1])})))
        
        for i in range(1000):
            sess.run(m["sy_op"])
        neg_data_seq.append(sess.run(m["sy_sample"]))
            
    d.set_neg([])
    plt.plot(neg_data_seq[-1][:,0], neg_data_seq[-1][:,1], ".")
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.savefig("figure/final.jpg")
    plt.show()
    
    print ("Real distribution")
    b = d.minibatch(d.get_size())
    plt.plot(b[0][:, 0], b[0][:, 1], '.')
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.savefig("figure/ref.jpg")
    plt.show()