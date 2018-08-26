import cntk
import numpy as np
import random as rng
from CtfConverter import CharMappings, convertToCTF

dir         = './text/'
fileName    = 'Shakespeare.txt'

timeSteps   = 1
timeShift   = 1 
outputSize  = 1
layers      = 2

lr          = 0.035
batchSize   = 256
maxEpochs   = 50
numClasses  = 255

def createNetwork(input, layers, numClasses):

    return cntk.layers.Sequential([        
        cntk.layers.For(range(layers), lambda: 
                   cntk.layers.Sequential([cntk.layers.Stabilizer(), cntk.layers.Recurrence(cntk.layers.LSTM(256), go_backwards=False)])),
        cntk.layers.Dropout(0.15),
        cntk.layers.Dense(numClasses)
    ])


def generateText(net, mapper, length):

    seed        = rng.randint(0, mapper.numClasses - 1)
    input       = np.zeros(timeSteps)

    input[timeSteps - 1]   = seed

    def process(output):
        return np.argmax(output, axis=2)[0,0]

    seq = mapper.toChar(seed)

    firstSeq = True

    for i in range(length):
        netIn       = cntk.one_hot(input, mapper.numClasses).eval()
        arguments   = ([netIn], [firstSeq])

        netOut  = net.eval(arguments)
        outNum  = process(netOut)

        seq += mapper.toChar(outNum)

        input       = np.roll(input, -1)
        input[timeSteps - 1]   = outNum
        firstSeq    = False

    return seq

from DataReader import loadData

# TODO: Look into Attention based models 
# i.e. something like cntk.layers.attention
# https://arxiv.org/pdf/1502.03044.pdf
# https://towardsdatascience.com/memory-attention-sequences-37456d271992
#
def trainNetwork():
    
    

    #convertToCTF(dir + fileName, './data/Shakespeare', timeSteps, timeShift, (253,5000))
    mapper, gens = loadData(dir+fileName, './data/Shakespeare', batchSize, timeSteps, timeShift, False, (253,500))

    #mapper  = CharMappings(loc='./data/Shakespeare', load=True)

    # Input with dynamic sequence axis 
    # consisting of a matrix of [steps-in-time X number-of-possible-characters]
    inputSeqAxis = cntk.Axis('inputAxis')
    input   = cntk.sequence.input_variable((timeSteps, mapper.numClasses), sequence_axis=inputSeqAxis, name='input')


    model   = createNetwork(input, layers, mapper.numClasses) 

    label   = cntk.sequence.input_variable(mapper.numClasses, sequence_axis=inputSeqAxis, name='label') 

    z       = model(input)
    loss    = cntk.cross_entropy_with_softmax(z, label) 
    error   = cntk.classification_error(z, label)

    printer = cntk.logging.ProgressPrinter(tag='Training', num_epochs=maxEpochs)   

    learner = cntk.momentum_sgd(z.parameters, lr, 0.9, minibatch_size=batchSize)
    #learner = cntk.fsadagrad(model.parameters, lr=lr, minibatch_size=batchSize, momentum=0.9, unit_gain=True)
    trainer = cntk.Trainer(z, (loss, error), learner, [printer])

    cntk.logging.log_number_of_parameters(z)

    numMinibatch = mapper.samples // batchSize


    for epoch in range(maxEpochs):
        mask = [True]
        for mb in range(numMinibatch):
            X, Y = next(gens['train'])
            arguments = ({ input: X, label: Y }, mask)
            mask = [False]
            trainer.train_minibatch(arguments)

        trainer.summarize_training_progress()
        print(generateText(z, mapper, 100))


    #cntk.train.training_session(
    #    trainer=trainer,
    #    mb_source=trainingReader,
    #    mb_size=batchSize,
    #    model_inputs_to_streams=inputMap,
    #    max_samples=maxEpochs * mapper.samples * 100
    #    ).train()

    #generateText(model)

    

trainNetwork()