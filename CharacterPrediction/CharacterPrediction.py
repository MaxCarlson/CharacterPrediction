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

def createReader(path, isTraining, inputDim, numClasses):

    featureStream = cntk.io.StreamDef(field='X', shape=numClasses, is_sparse=True)
    labelStream   = cntk.io.StreamDef(field='Y', shape=numClasses, is_sparse=True)

    deserializer = cntk.io.CTFDeserializer(path, cntk.io.StreamDefs(features = featureStream, labels = labelStream))

    return cntk.io.MinibatchSource(deserializer, randomize=isTraining, 
                                   max_sweeps=cntk.io.INFINITELY_REPEAT if isTraining else 1)

def createNetwork(input, numClasses):

    with cntk.layers.default_options(initial_state = 0.1):
        n = cntk.layers.Recurrence(cntk.layers.LSTM(timeSteps))(input)
        n = cntk.sequence.last(n)
        n = cntk.layers.Dropout(0.4)(n)
        n = cntk.layers.Dense(numClasses)(n)
        return n

def genBatch(X, Y, dataSet):
    # TODO: Need to look into randomizing the data's order

    dataSize = len(X)

    def asBatch(data, start, count):
        part = []
        for i in range(start, start+count):
            part.append(data[i])
        return np.array(part)

    #for i in range(0, dataSize - batchSize, batchSize):
    #    yield asBatch(X, i, batchSize), asBatch(Y, i, batchSize)

    for i in range(0, len(X[dataSet]) - batchSize, batchSize):
        yield asBatch(X[dataSet], i, batchSize), asBatch(Y[dataSet], i, batchSize)


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
    data = loadData(dir+fileName, './data/Shakespeare', batchSize, timeSteps, timeShift, (253,5000))

    mapper  = CharMappings(loc='./data/Shakespeare', load=True)

    # Input with dynamic sequence axis 
    # consisting of numClasses length one-hot vectors
    inputSeqAxis = cntk.Axis('inputAxis')
    input   = cntk.sequence.input_variable(mapper.numClasses, sequence_axis=inputSeqAxis, name='input')

    #input   = cntk.sequence.input_variable(mapper.numClasses, is_sparse=True, name='input')

    model   = createNetwork(input, mapper.numClasses) 

    label   = cntk.input_variable(mapper.numClasses, dynamic_axes=model.dynamic_axes, name='label') 
    #label   = cntk.sequence.input_variable(mapper.numClasses, sequence_axis=inputSeqAxis,  name='label') 

    #z       = model(input)

    trainingReader  = createReader('./data/Shakespeare_train.ctf', True, timeSteps, mapper.numClasses)
    inputMap        = { input: trainingReader.streams.features, label: trainingReader.streams.labels }

    loss    = cntk.cross_entropy_with_softmax(model, label) 
    error   = cntk.cross_entropy_with_softmax(model, label) 
    printer = cntk.logging.ProgressPrinter(tag='Training', num_epochs=maxEpochs)   

    learner = cntk.fsadagrad(model.parameters, lr=lr, minibatch_size=batchSize, momentum=0.9, unit_gain=True)
    trainer = cntk.Trainer(model, (loss, error), learner, [printer])

    cntk.logging.log_number_of_parameters(model)

    numMinibatch = mapper.samples // batchSize

    #cntk.train.training_session(
    #    trainer=trainer,
    #    mb_source=trainingReader,
    #    mb_size=batchSize,
    #    model_inputs_to_streams=inputMap,
    #    max_samples=maxEpochs * mapper.samples * 100
    #    ).train()

    #X, Y   = loadData(dir + fileName, mapper, timeSteps, timeShift)
    #X1, Y1 = next(genBatch(X, Y, "test"))
    #outs   = model(X1)
    #outsA  = np.argmax(outs, 1)
    #res    = np.argmax(Y1, 1)

    ls = []
    er = []
    for epoch in range(maxEpochs):
        for mb in range(numMinibatch):
            data = trainingReader.next_minibatch(batchSize, input_map=inputMap)
            trainer.train_minibatch(data)
            ls.append(trainer.previous_minibatch_loss_average)
            er.append(trainer.previous_minibatch_evaluation_average)

        trainer.summarize_training_progress()



        result = sum(outsA == res)

        print(X1[0][0])
        print('\n')
        print(X1[1][0])
        print('\n')

        
        

        print("epoch: {}, loss: {:.3f}".format(epoch, trainer.previous_minibatch_loss_average))


    generateText(model)

    

trainNetwork()