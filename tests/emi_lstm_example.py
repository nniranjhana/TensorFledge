from __future__ import print_function
import os
import sys
import tensorflow as tf
import inject
import numpy as np
# To include edgeml in python path
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

# MI-RNN and EMI-RNN imports
from edgeml_tf.graph.rnn import EMI_DataPipeline
from edgeml_tf.graph.rnn import EMI_BasicLSTM
from edgeml_tf.trainer.emirnnTrainer import EMI_Trainer, EMI_Driver
import edgeml_tf.utils

# Network parameters for our LSTM + FC Layer
NUM_HIDDEN = 32
NUM_TIMESTEPS = 48
NUM_FEATS = 9
FORGET_BIAS = 1.0
NUM_OUTPUT = 6
USE_DROPOUT = True
KEEP_PROB = 0.75

# For dataset API
PREFETCH_NUM = 5
BATCH_SIZE = 32

# Number of epochs in *one iteration*
NUM_EPOCHS = 2 
# Number of iterations in *one round*. After each iteration,
# the model is dumped to disk. At the end of the current
# round, the best model among all the dumped models in the
# current round is picked up..
NUM_ITER = 2
# A round consists of multiple training iterations and a belief
# update step using the best model from all of these iterations
NUM_ROUNDS = 2

# A staging direcory to store models
MODEL_PREFIX = '/tmp/model-lstm'

# Loading the data
x_train, y_train = np.load('./HAR/48_16/x_train.npy'), np.load('./HAR/48_16/y_train.npy')
x_test, y_test = np.load('./HAR/48_16/x_test.npy'), np.load('./HAR/48_16/y_test.npy')
x_val, y_val = np.load('./HAR/48_16/x_val.npy'), np.load('./HAR/48_16/y_val.npy')

# BAG_TEST, BAG_TRAIN, BAG_VAL represent bag_level labels. These are used for the label update
# step of EMI/MI RNN
BAG_TEST = np.argmax(y_test[:, 0, :], axis=1)
BAG_TRAIN = np.argmax(y_train[:, 0, :], axis=1)
BAG_VAL = np.argmax(y_val[:, 0, :], axis=1)
NUM_SUBINSTANCE = x_train.shape[1]
print("x_train shape is:", x_train.shape)
print("y_train shape is:", y_train.shape)
print("x_test shape is:", x_val.shape)
print("y_test shape is:", y_val.shape)

# Define the linear secondary classifier
def createExtendedGraph(self, baseOutput, *args, **kwargs):
    W1 = tf.Variable(np.random.normal(size=[NUM_HIDDEN, NUM_OUTPUT]).astype('float32'), name='W1')
    B1 = tf.Variable(np.random.normal(size=[NUM_OUTPUT]).astype('float32'), name='B1')
    y_cap = tf.add(tf.tensordot(baseOutput, W1, axes=1), B1, name='y_cap_tata')
    self.output = y_cap
    self.graphCreated = True
    
def addExtendedAssignOps(self, graph, W_val=None, B_val=None):
    W1 = graph.get_tensor_by_name('W1:0')
    B1 = graph.get_tensor_by_name('B1:0')
    W1_op = tf.assign(W1, W_val)
    B1_op = tf.assign(B1, B_val)
    self.assignOps.extend([W1_op, B1_op])

def restoreExtendedGraph(self, graph, *args, **kwargs):
    y_cap = graph.get_tensor_by_name('y_cap_tata:0')
    self.output = y_cap
    self.graphCreated = True
    
def feedDictFunc(self, keep_prob, **kwargs):
    feedDict = {self._emiGraph.keep_prob: keep_prob}
    return feedDict
    
EMI_BasicLSTM._createExtendedGraph = createExtendedGraph
EMI_BasicLSTM._restoreExtendedGraph = restoreExtendedGraph
EMI_BasicLSTM.addExtendedAssignOps = addExtendedAssignOps

if USE_DROPOUT is True:
    EMI_Driver.feedDictFunc = feedDictFunc

tf.reset_default_graph()

inputPipeline = EMI_DataPipeline(NUM_SUBINSTANCE, NUM_TIMESTEPS, NUM_FEATS, NUM_OUTPUT)
emiLSTM = EMI_BasicLSTM(NUM_SUBINSTANCE, NUM_HIDDEN, NUM_TIMESTEPS, NUM_FEATS,
                        forgetBias=FORGET_BIAS, useDropout=USE_DROPOUT)
emiTrainer = EMI_Trainer(NUM_TIMESTEPS, NUM_OUTPUT, lossType='xentropy')

# Construct the graph
g1 = tf.Graph()    
with g1.as_default():
    x_batch, y_batch = inputPipeline()
    y_cap = emiLSTM(x_batch)
    emiTrainer(y_cap, y_batch)
    
with g1.as_default():
    emiDriver = EMI_Driver(inputPipeline, emiLSTM, emiTrainer)

emiDriver.initializeSession(g1)
y_updated, modelStats = emiDriver.run(numClasses=NUM_OUTPUT, x_train=x_train,
                                      y_train=y_train, bag_train=BAG_TRAIN,
                                      x_val=x_val, y_val=y_val, bag_val=BAG_VAL,
                                      numIter=NUM_ITER, keep_prob=KEEP_PROB,
                                      numRounds=NUM_ROUNDS, batchSize=BATCH_SIZE,
                                      numEpochs=NUM_EPOCHS, modelPrefix=MODEL_PREFIX,
                                      fracEMI=0.5, updatePolicy='top-k', k=1)

def earlyPolicy_minProb(instanceOut, minProb, **kwargs):
    assert instanceOut.ndim == 2
    classes = np.argmax(instanceOut, axis=1)
    prob = np.max(instanceOut, axis=1)
    index = np.where(prob >= minProb)[0]
    if len(index) == 0:
        assert (len(instanceOut) - 1) == (len(classes) - 1)
        return classes[-1], len(instanceOut) - 1
    index = index[0]
    return classes[index], index


k = 2
predictions, predictionStep = emiDriver.getInstancePredictions(x_test, y_test, earlyPolicy_minProb,
                                                               minProb=0.99, keep_prob=1.0)
bagPredictions = emiDriver.getBagPredictions(predictions, minSubsequenceLen=k, numClass=NUM_OUTPUT)
print('Accuracy at k = %d before injections: %f' % (k,  np.mean((bagPredictions == BAG_TEST).astype(int))))

sess = emiDriver.getCurrentSession()
inject.inject(sess)

v = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
print(v)

predictions, predictionStep = emiDriver.getInstancePredictions(x_test, y_test, earlyPolicy_minProb,
                                                               minProb=0.99, keep_prob=1.0)
bagPredictions = emiDriver.getBagPredictions(predictions, minSubsequenceLen=k, numClass=NUM_OUTPUT)
print('Accuracy at k = %d after injections: %f' % (k,  np.mean((bagPredictions == BAG_TEST).astype(int))))
