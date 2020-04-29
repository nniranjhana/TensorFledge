# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import helpermethods
import tensorflow as tf
import numpy as np
import sys
from edgeml_tf.trainer.bonsaiTrainer import BonsaiTrainer
from edgeml_tf.graph.bonsai import Bonsai


def main():
    # Fixing seeds for reproducibility
    tf.set_random_seed(42)
    np.random.seed(42)

    # Hyper Param pre-processing
    args = helpermethods.getArgs()

    # Set 'isRegression' to be True, for regression. Default is 'False'.
    isRegression = args.regression

    sigma = args.sigma
    depth = args.depth

    projectionDimension = args.proj_dim
    regZ = args.rZ
    regT = args.rT
    regW = args.rW
    regV = args.rV

    totalEpochs = args.epochs

    learningRate = args.learning_rate

    dataDir = args.data_dir

    outFile = args.output_file

    (dataDimension, numClasses, Xtrain, Ytrain, Xtest, Ytest,
     mean, std) = helpermethods.preProcessData(dataDir, isRegression)

    sparZ = args.sZ

    if numClasses > 2:
        sparW = 0.2
        sparV = 0.2
        sparT = 0.2
    else:
        sparW = 1
        sparV = 1
        sparT = 1

    if args.sW is not None:
        sparW = args.sW
    if args.sV is not None:
        sparV = args.sV
    if args.sT is not None:
        sparT = args.sT

    if args.batch_size is None:
        batchSize = np.maximum(100, int(np.ceil(np.sqrt(Ytrain.shape[0]))))
    else:
        batchSize = args.batch_size

    useMCHLoss = True

    if numClasses == 2:
        numClasses = 1

    X = tf.placeholder("float32", [None, dataDimension])
    Y = tf.placeholder("float32", [None, numClasses])
#    sigmaI = tf.placeholder(tf.float32)

    currDir = helpermethods.createTimeStampDir(dataDir)

    helpermethods.dumpCommand(sys.argv, currDir)
    helpermethods.saveMeanStd(mean, std, currDir)

    # numClasses = 1 for binary case
    bonsaiObj = Bonsai(numClasses, dataDimension,
                       projectionDimension, depth, sigma, isRegression)

    bonsaiTrainer = BonsaiTrainer(bonsaiObj,
                                  regW, regT, regV, regZ,
                                  sparW, sparT, sparV, sparZ,
                                  learningRate, X, Y, useMCHLoss, outFile)

    sess = tf.InteractiveSession()

    sess.run(tf.global_variables_initializer())

    bonsaiTrainer.train(batchSize, totalEpochs, sess,
                        Xtrain, Xtest, Ytrain, Ytest, dataDir, currDir)

    #score, X_ = bonsaiObj(X = Xtest, sigmaI = 1e9)
    #correctPrediction = tf.equal(tf.argmax(tf.transpose(score), 1), tf.argmax(Ytest, 1))
    #acc = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
    #print (sess.run(acc))

    acc = bonsaiTrainer.test(Xtest, Ytest)
    print(sess.run(acc))

#    print (sess.run(bonsaiTrainer.accuracy, feed_dict={X:Xtest, Y:Ytest, sigmaI:1e9}))

#    v = tf.all_variables()
#    print (v)
#    bOSigmaI = 1e9
#    tf.reset_default_graph()
#    sigmaI = tf.get_default_graph().get_tensor_by_name("sigmaI")
#    sigma = graph.get_tensor_by_name("sigma")
#    print (sess.run(bonsaiTrainer.accuracyGraph(), feed_dict={X:Xtest,Y:Ytest,sigmaI:bOSigmaI}))

    sess.close()
    sys.stdout.close()


if __name__ == '__main__':
    main()

