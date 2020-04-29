#!/usr/bin/python

import tensorflow as tf
from struct import pack, unpack
import numpy as np
import random
import yaml

def config(confFile = None):
    fiConf = {}
    if confFile == None:
        fiConf["Artifact"] = 0
        fiConf["Type"] = "shuffle"
        return fiConf
    try:
        fiConfs = open(confFile, "r")
    except IOError:
        print "Unable to open the config file ", confFile
        return fiConf
    if confFile.endswith(".yaml"):
        fiConf = yaml.load(fiConfs)
    else:
        print "Unsupported file format: ", confFile
    return fiConf

def bitflip(f, pos):
    f_ = pack('f', f)
    b = list(unpack('BBBB', f_))
    [q, r] = divmod(pos, 8)
    b[q] ^= 1 << r
    f_ = pack('BBBB', *b)
    f = unpack('f', f_)
    return f[0]

class inject():

    def __init__(self, sess, confFile="confFiles/sample.yaml"):

        fiConf = config(confFile)
        self.session = sess
        fiFunc = getattr(self, fiConf["Type"])
        fiFunc(sess, fiConf)

    def shuffle(self, sess, fiConf):
        v = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[fiConf["Artifact"]]
        v_ = tf.random_shuffle(v)
        sess.run(tf.assign(v, v_))

    def mutate(self, sess, fiConf):
        v = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[fiConf["Artifact"]]
        num = v.shape.num_elements()
        sz = fiConf["Amount"]
        ind = random.sample(range(num), sz)
        i, j = v.get_shape().as_list()
        ind_ = []
        for item in ind:
            ind_.append([item/j, item%j])
        upd = []
        if (fiConf["Bit"]=='N'):
            for item in ind_:
                val = v[item[0]][item[1]].eval(session=sess)
                pos = random.randint(0, 31)
                val_ = bitflip(val, pos)
                upd.append(val_)
        else:
            pos = fiConf["Bit"]
            for item in ind_:
                val = v[item[0]][item[1]].eval(session=sess)
                val_ = bitflip(val, pos)
                upd.append(val_)
        v_ = tf.scatter_nd_update(v, ind_, upd)
        sess.run(tf.assign(v, v_))

    def zeros(self, sess, fiConf):
        v = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[fiConf["Artifact"]]
        num = v.shape.num_elements()
        sz = (fiConf["Amount"] * num) / 100
        ind = random.sample(range(num), sz)
        i, j = v.get_shape().as_list()
        ind_ = []
        for item in ind:
            ind_.append([item/j, item%j])
        upd = tf.zeros([sz], tf.float32)
        v_ = tf.scatter_nd_update(v, ind_, upd)
        sess.run(tf.assign(v, v_))
