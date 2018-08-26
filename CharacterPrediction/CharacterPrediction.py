import cntk
import numpy as np
from CtfConverter import CharMappings, convertToCTF

dir         = './text/'
fileName    = 'Shakespeare.txt'

timeSteps   = 50
timeShift   = 1 
outputSize  = 1

lr          = 0.035
batchSize   = 256
maxEpochs   = 5
numClasses  = 255

def createNetwork(input, numClasses):

    return cntk.layers.Sequential([        
        cntk.layers.For(range(10), lambda: 
                   cntk.layers.Sequential([cntk.layers.Stabilizer(), cntk.layers.Recurrence(cntk.layers.LSTM(128), go_backwards=False)])),
        cntk.layers.Dense(numClasses)
    ])

    with cntk.layers.default_options(initial_state = 0.1):
        n = cntk.layers.Recurrence(cntk.layers.LSTM(timeSteps))(input)
        n = cntk.sequence.last(n)
        n = cntk.layers.Dropout(0.4)(n)
        n = cntk.layers.Dense(numClasses)(n)
        return n

def generateText(net):

    seqLen  = 100
    seedSeq = 'ENTER: '

    def strToNums(str):
        lst = []
        for c in str:
            lst.append(ord(c))
        return cntk.one_hot(np.array(lst), numClasses).eval()

    seq         = seedSeq
    masterSeq   = seedSeq

    for i in range(seqLen):
        out          = net(strToNums(seq))
        digit        = np.argmax(out)
        masterSeq   += chr(digit)
        seq         += chr(digit)

        if len(seq) >= timeSteps:
            seq = seq[1:]


    print('Seed Sequence: {}'.format(seedSeq))
    print('Output: {}'.format(masterSeq))
    return

from DataReader import loadData

# TODO: Look into Attention based models 
# i.e. something like cntk.layers.attention
# https://arxiv.org/pdf/1502.03044.pdf
# https://towardsdatascience.com/memory-attention-sequences-37456d271992
#
def trainNetwork():
    
    

    #convertToCTF(dir + fileName, './data/Shakespeare', timeSteps, timeShift, (253,5000))
    mapper, gens = loadData(dir+fileName, './data/Shakespeare', batchSize, timeSteps, timeShift, True, (253,500))

    #mapper  = CharMappings(loc='./data/Shakespeare', load=True)

    # Input with dynamic sequence axis 
    # consisting of numClasses length one-hot vectors
    inputSeqAxis = cntk.Axis('inputAxis')
    input   = cntk.sequence.input_variable((timeSteps, mapper.numClasses), sequence_axis=inputSeqAxis, name='input')

    #input   = cntk.sequence.input_variable(mapper.numClasses, name='input')

    model   = createNetwork(input, mapper.numClasses) 

    #label   = cntk.input_variable(mapper.numClasses, dynamic_axes=model.dynamic_axes, name='label') 
    label   = cntk.sequence.input_variable(mapper.numClasses, sequence_axis=inputSeqAxis, name='label') 

    z       = model(input)

    #trainingReader  = createReader('./data/Shakespeare_train.ctf', True, timeSteps, mapper.numClasses)
    #inputMap        = { input: trainingReader.streams.features, label: trainingReader.streams.labels }

    loss    = cntk.cross_entropy_with_softmax(z, label) 
    #error   = cntk.cross_entropy_with_softmax(z, label) 
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
            arguments = ({ z.arguments[0]: X, label: Y }, mask)
            mask = [False]
            trainer.train_minibatch(arguments)

        trainer.summarize_training_progress()

    #cntk.train.training_session(
    #    trainer=trainer,
    #    mb_source=trainingReader,
    #    mb_size=batchSize,
    #    model_inputs_to_streams=inputMap,
    #    max_samples=maxEpochs * mapper.samples * 100
    #    ).train()

    generateText(model)

    

trainNetwork()