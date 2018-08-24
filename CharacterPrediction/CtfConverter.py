import cntk
import numpy as np
import pickle

# Create a custom mapping of charcters in the file
# so in the likely case we're not using all ASCII characters
# we don't need to create one-hot vectors through all of ASCII spce
# but instead just map the charcters we're using
class CharMappings():
    def __init__(self, lines, loc, excludeChars):
        self.loc        = loc
        self.exclude    = excludeChars
        self.charToInt  = dict()
        self.intToChar  = dict()
        self.mapChars(lines)
       
    def mapChars(self, lines):
        i = 0
        for l in lines:
            for c in l:
                if c not in self.charToInt and c not in self.exclude: # Exclude needs to also be removed from all feature inputs
                    self.charToInt[c] = i
                    i += 1

        for k, v in self.charToInt.items():
            self.intToChar[v] = k

    def toNum(self, ch):
        return self.charToInt[ch]
    def toChar(self, nm):
        return self.intToChar[nm]

    def save(self):




def convertToCTF(filePath, dest, timeSteps, timeShift, lineShape, excludeChars = None):

    file    = open(filePath)
    lines   = file.readlines()[lineShape[0]:lineShape[1]]
    file.close()

    # Create and save custom character mapper
    mapper = CharMappings(lines, dest, excludeChars)

    pickeld = pickle.dumps(mapper)

    lld = pickle.loads(pickeld)

    # Create and save file containing features and labels
    # in CTF format with custom mappings

    return


def splitData(data, valSize = 0.1, testSize = 0.1):

    length      = len(data)
    trainLen    = int((1.0 - valSize - testSize) * length)
    valLen      = int(valSize  * length)
    testLen     = int(testSize * length)


    train, val, test = data[0:trainLen], data[trainLen:trainLen+valLen], data[trainLen+valLen:]
    train   = cntk.one_hot(train, numClasses).eval(device=cntk.cpu()) # For the moment this gets around the memory limitation thrown up by my GPU, but we really need to generate this data beforehand!
    val     = cntk.one_hot(val, numClasses).eval()
    test    = cntk.one_hot(test, numClasses).eval()

    return {"train": train, "val": val, "test": test}

# TODO: MUST reduce size of numClasses. No need to use all 255 ascii values!!!!
def loadData(path, timeSteps, timeShift):
    
    file    = open(path)
    lines   = file.readlines()[253:124437] # Skip the header lines in the Shakespeare file
    file.close()

    # Pair lines down more for expedient testing early on
    lines = lines[0:5000]
    
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

    return splitData(X), splitData(Y)