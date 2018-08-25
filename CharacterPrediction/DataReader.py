import cntk
import numpy as np
import random as rng
from CtfConverter import CharMappings

# TODO: Add support for files too big for memory if needed!
# TODO: Add in more randomized method of picking than just random start  points
def generator(batchSize, timeSteps, mapper, dest, dataName):

    data    = np.loadtxt(dest + '_' + dataName + '.ctf', delimiter=' ')
    X       = data[:, 0:timeSteps]
    Y       = data[:, timeSteps:]

    X = cntk.one_hot(X, mapper.numClasses).eval()
    Y = cntk.one_hot(Y, mapper.numClasses).eval()

    size = len(X)
    while True:
        # TODO: Need edge checking here!!!!!!!
        start   = rng.randint(0, size)
        end     = start + batchSize

        yield X[start:end], Y[start:end]

def writeToFile(dest, mapper, length, timeSteps, timeShift, data, dataName):

    file = open(dest + '_' + dataName + '.ctf', "w+")

    # File format is:
    # All numbers on a line except the last are the input (in non-sparse format), 
    # the last on a line is the index of the expected output

    # TODO: Add variable length time steps with padding in here!
    for i in range(length - timeSteps - timeShift + 1):
        fStr = str(data[i:i + timeSteps]).replace(',', '')[1:-1] + ' '
        lStr = str(data[timeShift + timeSteps + i - 1]).replace(',','')

        file.write(fStr)
        file.write(lStr)

        file.write('\n')

    return

def loadData(filePath, dest, batchSize, timeSteps, timeShift, load=False, lineShape = [0,-1], split = [0.90, 0.1]):
    file    = open(filePath, "r")
    lines   = file.readlines()[lineShape[0]:lineShape[1]]
    file.close()

    mapper = None
    if load == False:
        mapper  = CharMappings(lines, dest)

        data = []
        for l in lines:
            for c in l:
                data.append(mapper.toNum(c))

        # Create sepperate files for 
        # training and validation datasets
        trainLen = int(len(data) * split[0])
        validLen = int(len(data) * split[1])

        writeToFile(dest, mapper, trainLen, timeSteps, timeShift, data[0:trainLen], 'train')
        writeToFile(dest, mapper, validLen, timeSteps, timeShift, data[trainLen: ], 'validation')
    else:
        mapper = CharMappings(loc=dest, load=True)


    gens = { 'train': generator(batchSize, timeSteps, mapper, dest, 'train'), 
            'validation': generator(batchSize, timeSteps, mapper, dest, 'validation') }

    return mapper, gens