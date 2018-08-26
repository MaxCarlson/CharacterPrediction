import cntk
import numpy as np
import random as rng
from DataReader import loadData
from CtfConverter import CharMappings, convertToCTF

dir         = './text/'
fileName    = 'tinyshakespeare.txt'

timeSteps   = 1
timeShift   = 1 
outputSize  = 1
layers      = 2

lr          = 0.035
batchSize   = 512
maxEpochs   = 50

def createNetwork(input, layers, numClasses):

    return cntk.layers.Sequential([        
        cntk.layers.For(range(layers), lambda: 
                   cntk.layers.Sequential([cntk.layers.Stabilizer(), cntk.layers.Recurrence(cntk.layers.LSTM(256), go_backwards=False)])),
        cntk.layers.Dense(numClasses)
    ])


def generateText(net, mapper, length):

    seed        = rng.randint(0, mapper.numClasses - 1)
    input       = np.zeros(timeSteps)

    input[timeSteps - 1]   = seed

    def process(output):
        return np.argmax(output, axis=2)[0,0]

    seq = mapper.toChar(seed)

    startSeq = True

    for i in range(length):
        netIn       = cntk.one_hot(input, mapper.numClasses).eval()
        arguments   = ([netIn], [startSeq])

        netOut  = net.eval(arguments)
        outNum  = process(netOut)

        seq += mapper.toChar(outNum)

        input       = np.roll(input, -1)
        input[timeSteps - 1]   = outNum
        startSeq    = False

    return seq

# TODO: Feed with this to test
def get_data(p, minibatch_size, data, mapper):
    # the character LM predicts the next character so get sequences offset by 1
    xi = [mapper.toNum(ch) for ch in data[p:p+minibatch_size]]
    yi = [mapper.toNum(ch) for ch in data[p+1:p+minibatch_size+1]]
    
    # a slightly inefficient way to get one-hot vectors but fine for low vocab (like char-lm)
    X = np.eye(mapper.numClasses, dtype=np.float32)[xi]
    Y = np.eye(mapper.numClasses, dtype=np.float32)[yi]

    X = np.expand_dims(X, 1)

    # return a list of numpy arrays for each of X (features) and Y (labels)
    return [X], [Y]

# TODO: Look into Attention based models 
# i.e. something like cntk.layers.attention
# https://arxiv.org/pdf/1502.03044.pdf
# https://towardsdatascience.com/memory-attention-sequences-37456d271992
#
def trainNetwork():
    
    mapper, gens = loadData(dir+fileName, './data/Shakespeare', batchSize, timeSteps, timeShift, load=True, lineShape=(0,40000))

    datal = open(dir+fileName, "r").readlines()
    data = open(dir+fileName, "r").read()
    mapper2 = CharMappings(datal, './data/Shakespeare')

    X1, Y1 = get_data(0, batchSize, data, mapper2)
    X2, Y2 = next(gens['train'])

    X3 = X1 == X2
    Y3 = Y1 == Y2

    # Input with dynamic sequence axis 
    # consisting of a matrix of [steps-in-time X number-of-possible-characters]
    inputSeqAxis = cntk.Axis('inputAxis')
    input   = cntk.sequence.input_variable((timeSteps, mapper.numClasses), sequence_axis=inputSeqAxis, name='input')


    model   = createNetwork(input, layers, mapper.numClasses) 

    label   = cntk.sequence.input_variable(mapper.numClasses, sequence_axis=inputSeqAxis, name='label') 

    z       = model(input)
    loss    = cntk.cross_entropy_with_softmax(z, label) 
    error   = cntk.classification_error(z, label)

    printer = cntk.logging.ProgressPrinter(tag='Training', freq=100, num_epochs=maxEpochs)   

    lr_per_sample = cntk.learning_parameter_schedule_per_sample(0.001)
    momentum_schedule = cntk.momentum_schedule_per_sample(0.9990913221888589)
    learner = cntk.momentum_sgd(z.parameters, lr_per_sample, momentum_schedule,
                           gradient_clipping_threshold_per_sample=5.0,
                           gradient_clipping_with_truncation=True)

    #learner = cntk.momentum_sgd(z.parameters, lr, 0.9, minibatch_size=batchSize)
    #learner = cntk.fsadagrad(model.parameters, lr=lr, minibatch_size=batchSize, momentum=0.9, unit_gain=True)
    trainer = cntk.Trainer(z, (loss, error), learner, [printer])

    numMinibatch = mapper.samples // batchSize

    print("Input sequence length: {}; unique characters {};".format(timeSteps, mapper.numClasses))
    cntk.logging.log_number_of_parameters(z)
    print("Datset size {}; {} Epochs; {} minibatches per epoch".format(mapper.samples, maxEpochs, numMinibatch))

    for epoch in range(maxEpochs):
        mask = [True]
        for mb in range(numMinibatch):
            X, Y = next(gens['train'])
            #X, Y = get_data(mb, batchSize, data, mapper)
            arguments = ({ input: X, label: Y }, mask)
            mask = [False]
            trainer.train_minibatch(arguments)

            if mb % 100 == 0:
                print(generateText(z, mapper, 200) + '\n')

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