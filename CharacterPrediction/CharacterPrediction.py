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
numClasses  = 255


def splitData(data, valSize = 0.1, testSize = 0.1):

    length      = len(data)
    trainLen    = int((1.0 - valSize - testSize) * length)
    valLen      = int(valSize  * length)
    testLen     = int(testSize * length)


    train, val, test = data[0:trainLen], data[trainLen:trainLen+valLen], data[trainLen+valLen:]

    return {"train": train, "val": val, "test": test}

def loadData(path, timeSteps, timeShift):
    
    file    = open(path)
    lines   = file.readlines()[253:124437] # Skip the header lines in the Shakespeare file
    file.close()

    # Pair lines down more for expedient testing early on
    lines = lines[0:1000]
    
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

    return X, Y#splitData(X), splitData(Y)

def createNetwork(input):

    with cntk.layers.default_options(initial_state = 0.1):
        n = cntk.layers.Recurrence(cntk.layers.LSTM(timeSteps))(input)
        n = cntk.sequence.last(n)
        n = cntk.layers.Dropout(0.4)(n)
        n = cntk.layers.Dense(numClasses)(n)
        return n

def genBatch(X, Y, dataSet):
    # TODO: Need to look into randomizing the data's order

    def asBatch(data, start, count):
        part = []
        for i in range(start, start+count):
            part.append(data[i])
        return np.array(part)

    for i in range(0, len(X) - batchSize, batchSize):
        yield asBatch(X, i, batchSize), asBatch(Y, i, batchSize)

    #for i in range(0, len(X[dataSet]) - batchSize, batchSize):
    #    yield asBatch(X[dataSet], i, batchSize), asBatch(Y[dataSet], i, batchSize)

# TODO: Look into Attention based models 
# i.e. something like cntk.layers.attention
# https://arxiv.org/pdf/1502.03044.pdf
# https://towardsdatascience.com/memory-attention-sequences-37456d271992
#
def trainNetwork():
    
    #xAxes = [cntk.Axis.default_batch_axis(), cntk.Axis.default_dynamic_axis()]
    #input = cntk.input_variable(1, dynamic_axes=xAxes)
    input   = cntk.sequence.input_variable(numClasses)

    X, Y    = loadData(dir + fileName, timeSteps, timeShift)

    X = cntk.one_hot(X, numClasses)
    Y = cntk.one_hot(Y, numClasses)

    model   = createNetwork(input)

    label   = cntk.input_variable(numClasses, dynamic_axes=model.dynamic_axes, name='label')

    loss    = cntk.cross_entropy_with_softmax(model, label) #cntk.squared_error(model, label)
    error   = cntk.cross_entropy_with_softmax(model, label) #cntk.squared_error(model, label)
    printer = cntk.logging.ProgressPrinter(tag='Training', num_epochs=maxEpochs)   

    learner = cntk.fsadagrad(model.parameters, lr=lr, minibatch_size=batchSize, momentum=0.9, unit_gain=True)
    trainer = cntk.Trainer(model, (loss, error), [learner, printer])


    for epoch in range(maxEpochs):
        for X1, Y1 in genBatch(X, Y, "train"):
            trainer.train_minibatch({X: X1, Y: Y1})
        print("epoch: {}, loss: {:.3f}".format(epoch, trainer.previous_minibatch_loss_average))

    

trainNetwork()