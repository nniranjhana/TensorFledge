from __future__ import print_function
import os
import sys
import tensorflow as tf
import inject
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] ='1'

# FastGRNN and FastRNN imports
from edgeml_tf.graph.rnn import EMI_DataPipeline
from edgeml_tf.graph.rnn import EMI_FastGRNN
from edgeml_tf.graph.rnn import EMI_FastRNN
from edgeml_tf.trainer.emirnnTrainer import EMI_Trainer, EMI_Driver
import edgeml_tf.utils

# Network parameters for our FastGRNN + FC Layer
NUM_HIDDEN = 16
NUM_TIMESTEPS = 48
NUM_FEATS = 9
FORGET_BIAS = 1.0
NUM_OUTPUT = 6
USE_DROPOUT = False
KEEP_PROB = 0.9

# Non-linearities can be chosen among "tanh, sigmoid, relu, quantTanh, quantSigm"
UPDATE_NL = "quantTanh"
GATE_NL = "quantSigm"

# Ranks of Parameter matrices for low-rank parameterisation to compress models.
WRANK = 5
URANK = 6

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
MODEL_PREFIX = '/tmp/model-fgrnn'



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

def restoreExtendedGraph(self, graph, *args, **kwargs):
    y_cap = graph.get_tensor_by_name('y_cap_tata:0')
    self.output = y_cap
    self.graphCreated = True
    
def feedDictFunc(self, keep_prob=None, inference=False, **kwargs):
    if inference is False:
        feedDict = {self._emiGraph.keep_prob: keep_prob}
    else:
        feedDict = {self._emiGraph.keep_prob: 1.0}
    return feedDict

    
EMI_FastGRNN._createExtendedGraph = createExtendedGraph
EMI_FastGRNN._restoreExtendedGraph = restoreExtendedGraph
if USE_DROPOUT is True:
    EMI_FastGRNN.feedDictFunc = feedDictFunc

inputPipeline = EMI_DataPipeline(NUM_SUBINSTANCE, NUM_TIMESTEPS, NUM_FEATS, NUM_OUTPUT)
emiFastGRNN = EMI_FastGRNN(NUM_SUBINSTANCE, NUM_HIDDEN, NUM_TIMESTEPS, NUM_FEATS, wRank=WRANK, uRank=URANK, 
                           gate_non_linearity=GATE_NL, update_non_linearity=UPDATE_NL, useDropout=USE_DROPOUT)
emiTrainer = EMI_Trainer(NUM_TIMESTEPS, NUM_OUTPUT, lossType='xentropy')

tf.reset_default_graph()
g1 = tf.Graph()    
with g1.as_default():
    # Obtain the iterators to each batch of the data
    x_batch, y_batch = inputPipeline()
    # Create the forward computation graph based on the iterators
    y_cap = emiFastGRNN(x_batch)
    # Create loss graphs and training routines
    emiTrainer(y_cap, y_batch)



with g1.as_default():
    emiDriver = EMI_Driver(inputPipeline, emiFastGRNN, emiTrainer)

emiDriver.initializeSession(g1)
y_updated, modelStats = emiDriver.run(numClasses=NUM_OUTPUT, x_train=x_train,
                                      y_train=y_train, bag_train=BAG_TRAIN,
                                      x_val=x_val, y_val=y_val, bag_val=BAG_VAL,
                                      numIter=NUM_ITER, keep_prob=KEEP_PROB,
                                      numRounds=NUM_ROUNDS, batchSize=BATCH_SIZE,
                                      numEpochs=NUM_EPOCHS, modelPrefix=MODEL_PREFIX,
                                      fracEMI=0.5, updatePolicy='top-k', k=1)


# Early Prediction Policy: We make an early prediction based on the predicted classes
#     probability. If the predicted class probability > minProb at some step, we make
#     a prediction at that step.
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

def getEarlySaving(predictionStep, numTimeSteps, returnTotal=False):
    predictionStep = predictionStep + 1
    predictionStep = np.reshape(predictionStep, -1)
    totalSteps = np.sum(predictionStep)
    maxSteps = len(predictionStep) * numTimeSteps
    savings = 1.0 - (totalSteps / maxSteps)
    if returnTotal:
        return savings, totalSteps
    return savings

k = 2
predictions, predictionStep = emiDriver.getInstancePredictions(x_test, y_test, earlyPolicy_minProb, minProb=0.99)
bagPredictions = emiDriver.getBagPredictions(predictions, minSubsequenceLen=k, numClass=NUM_OUTPUT)
print('Accuracy at k = %d before injections: %f' % (k,  np.mean((bagPredictions == BAG_TEST).astype(int))))
print('Additional savings: %f' % getEarlySaving(predictionStep, NUM_TIMESTEPS))

sess = emiDriver.getCurrentSession()
inject.inject(sess)

v = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
print(v)

k = 2
predictions, predictionStep = emiDriver.getInstancePredictions(x_test, y_test, earlyPolicy_minProb, minProb=0.99)
bagPredictions = emiDriver.getBagPredictions(predictions, minSubsequenceLen=k, numClass=NUM_OUTPUT)
print('Accuracy at k = %d after injections: %f' % (k,  np.mean((bagPredictions == BAG_TEST).astype(int))))
print('Additional savings: %f' % getEarlySaving(predictionStep, NUM_TIMESTEPS))

