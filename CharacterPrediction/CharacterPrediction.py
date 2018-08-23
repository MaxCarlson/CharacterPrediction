import cntk
import numpy as np


dir         = './text/'
fileName    = 'Shakespeare.txt'


timeSteps   = 5 # Can these by dynamic?
timeShift   = 1 
outputSize  = 1

lr          = 0.02
batchSize   = 64
maxEpochs   = 100

def loadData(path, timeSteps, timeShift):
    
    file    = open(path)
    lines   = file.readlines()[253:124437] # Skip the header lines in the Shakespeare file
    file.close()

    # Pair lines down more for expedient testing early on
    lines = lines[0:10000]
    
    # Convert the character data into numbers
    data = []
    for l in lines:
        for c in l:
            data.append(ord(c))
    
    a = len(data)
    
    X = []
    Y = []

    # Create our inputs. Make sure they're within realistic bounds
    for i in range(len(data) - timeSteps - timeShift + 1):
        X.append(np.array(data[i:i + timeSteps]))

    X = np.array(X)

    Y = np.array(data[timeShift + timeSteps - 1 :])

    return X, Y

def createNetwork(input):

    with cntk.layers.default_options(initial_state = 0.1):
        n = cntk.layers.Recurrence(cntk.layers.LSTM(timeSteps))(input)
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

    X, Y    = loadData(dir + fileName, timeSteps, timeShift)

    model   = createNetwork(input)

    label   = cntk.input_variable(1, dynamic_axes=model.dynamic_axes, name='label')

    loss    = cntk.cross_entropy_with_softmax(model, label) #cntk.squared_error(model, label)
    error   = cntk.cross_entropy_with_softmax(model, label) #cntk.squared_error(model, label)
    printer = cntk.logging.ProgressPrinter(tag='Training', num_epochs=maxEpochs)   

    learner = cntk.fsadagrad(model.parameters, lr=lr, minibatch_size=batchSize, momentum=0.9, unit_gain=True)
    trainer = cntk.Trainer(model, (loss, error), [learner, printer])


    for epoch in range(maxEpochs):
        for X1, Y1 in genBatch():
            pass

    

trainNetwork()