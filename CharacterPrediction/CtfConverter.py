import cntk
import numpy as np
import pickle
import copy

# Create a custom mapping of charcters in the file
# so in the likely case we're not using all ASCII characters
# we don't need to create one-hot vectors through all of ASCII spce
# but instead just map the charcters we're using
class CharMappings():
    def __init__(self, lines = None, loc = './mappings/', excludeChars = {}, load = False):
        self.loc        = loc
        self.exclude    = excludeChars
        self.charToInt  = dict()
        self.intToChar  = dict()
        self.numClasses = 0
        self.samples    = 0
        
        if lines:
            self.mapChars(lines)
            self.save()
        elif load:
            self.load()
       
    def mapChars(self, lines):
        i = 0
        for l in lines:
            for c in l:
                if c not in self.charToInt and c not in self.exclude: 
                    self.charToInt[c] = i
                    i += 1
                self.samples += 1

        for k, v in self.charToInt.items():
            self.intToChar[v] = k

        self.numClasses = len(self.charToInt)

    def toNum(self, ch):
        return self.charToInt[ch]

    def toChar(self, nm):
        return self.intToChar[nm]

    def save(self):
        pickle.dump(self, open(self.loc + '_map.bin', "wb"))

    def load(self):
        l = pickle.load(open('./data/Shakespeare_map.bin', "rb"))
        self.__dict__.update(l.__dict__)

    def __len__(self):
        return self.samples

def writeToFile(dest, mapper, length, timeSteps, timeShift, data, setName, featName, labelName):
    
    file = open(dest + '_' + setName + '.ctf', "w+")

    # Create and save file containing features and labels
    # in CTF format with custom mappings
    for i in range(length - timeSteps - timeShift + 1):
        fStr = str(data[i:i + timeSteps]).replace(',', '')[1:-1] + ' '
        fStr = fStr.replace(' ', ':1 ')

        lStr = str(data[timeShift + timeSteps + i - 1]).replace(',','') + ':1'


        file.write('|' + featName  + ' ' + fStr)
        file.write('|' + labelName + ' ' + lStr)

        file.write('\n')

def convertToCTF(filePath, dest, timeSteps, timeShift, lineShape, split = [0.8, 0.1, 0.1], excludeChars = {}):

    file    = open(filePath, "r")
    lines   = file.readlines()[lineShape[0]:lineShape[1]]
    file.close()

    # Create and save custom character mapper
    mapper  = CharMappings(lines, dest, excludeChars)

    # Convert to custom mapping array
    # TODO: If file is too large for memory this won't work
    data = []
    for l in lines:
        for c in l:
            data.append(mapper.toNum(c))

    # Create sepperate files for training, test
    # and validation datasets
    trainLen = int(len(data) * split[0])
    validLen = int(len(data) * split[1])
    testLen  = int(len(data) * split[2])

    writeToFile(dest, mapper, trainLen, timeSteps, timeShift, data[0:trainLen],                 'train',      'X', 'Y')
    writeToFile(dest, mapper, validLen, timeSteps, timeShift, data[trainLen:trainLen+validLen], 'validation', 'X', 'Y')
    writeToFile(dest, mapper, testLen,  timeSteps, timeShift, data[trainLen+validLen: ],        'test',       'X', 'Y')

