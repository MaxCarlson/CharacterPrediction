import cntk
import numpy as np


dir         = './text/'
fileName    = 'Shakespeare.txt'


inputSize   = 5 # Can these by dynamic?
outputDist  = 1
outputSize  = 1

lr          = 0.02
batchSize   = 64
maxEpochs   = 100

def loadData(path):
    pass

def createNetwork(input):

    with cntk.layers.default_options(initial_state = 0.1):
        n = cntk.layers.Recurrence(cntk.layers.LSTM(inputSize))(input)
        n = cntk.sequence.last(n)
        n = cntk.layers.Dropout(0.4)(n)
        n = cntk.layers.Dense(outSize)(n)
        return n

def genBatch(X, Y):
    pass

def trainNetwork():

    #xAxes = [cntk.Axis.default_batch_axis(), cntk.Axis.default_dynamic_axis()]
    #input = cntk.input_variable(1, dynamic_axes=xAxes)
    input   = cntk.sequence.input_variable(1)

    X, Y    = loadData(dir + fileName)

    model   = createNetwork(input)

    label   = cntk.input_variable(1, dynamic_axes=model.dynamic_axes, name='label')

    loss    = cntk.squared_error(model, label)
    error   = cntk.squared_error(model, label)
    printer = cntk.logging.ProgressPrinter(tag='Training', num_epochs=maxEpochs)   

    learner = cntk.fsadagrad(model.parameters, lr=lr, minibatch_size=batchSize, momentum=0.9, unit_gain=True)
    trainer = cntk.Trainer(model, (loss, error), [learner, printer])


    for epoch in range(maxEpochs):
        for X1, Y1 in genBatch()

    pass