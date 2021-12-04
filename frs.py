import numpy as np


def getSimilarity(x, y, kernelParameter=0.15):
    """

    :param x: sample x 
    :param y: sample y 
    :param kernelParameter: the parameter of Gaussin kernel
    :return: the similarity between x and y
    """
    x, y = np.array(x), np.array(y)
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * kernelParameter))


def getLowerApproximation(sampleData, perLabel, data, label, k=1, weightDict=None):
    """

    :param k: the num of neighbours to be weighted 
    :param weightDict: samples weight
    :param sampleData: one sample data
    :param perLabel: one specific class to be computed 
    :param data: 
    :param label: 
    :return: the lower approximation of one sample to the specific class (perLabel)
    """
    # the nearest k samples
    sampleNum = data.shape[0]
    if weightDict is None:
        weightDict = {i: 1 for i in range(len(data))}
    dissimilarity = {i: 1 - getSimilarity(sampleData, data[i]) for i in range(sampleNum) if
                     label[i] != perLabel and np.sum(data[i] != sampleData) != 0}
    if len(dissimilarity) == 0:
        return -1
    dissimilarity = dict(sorted(dissimilarity.items(), key=lambda x: x[1]))
    # sort the similarity list
    candidateKeys = list(dissimilarity.keys())[:k]
    # get the first k samples
    if k == 0:
        # FLA
        return list(dissimilarity.values())[0]
    if k == 1:
        # WFLA
        tempMin = np.sum([weightDict[i] * dissimilarity[i]
                         for i in candidateKeys])
    else:
        # RWFLA
        tempMin = np.sum([weightDict[i] * dissimilarity[i] for i in candidateKeys]) / np.sum(
            [weightDict[i] for i in candidateKeys])

    return tempMin


def getPositiveRegion(sampleData, data, label, weightDict=None, k=1, uniqueLabel=None):
    """

    :param uniqueLabel: label list without duplicates
    :param k: same with above
    :param weightDict: default None
    :param sampleData: same with above
    :param data: 
    :param label:
    :return: get the positive region of one sample
    """
    if uniqueLabel is not None:
        labelUnique = uniqueLabel
    else:
        labelUnique = np.unique(label)

    tempMax = []
    for perLabel in labelUnique:
        tempMax.append(
            getLowerApproximation(sampleData, perLabel, data, label, weightDict=weightDict, k=k))

    return max(tempMax), np.array(tempMax)


def getDependence(data, label, weightDict=None, k=1):
    """

    :param k:
    :param badPoints: index of badPoints
    :param weightDict: 
    :param data: one modal-data
    :param label:
    :return: return dependence
    """
    sampleNum = data.shape[0]
    tempSum = 0
    for sampleIdx in range(sampleNum):
        tempSum += getPositiveRegion(data[sampleIdx],
                                     data, label, weightDict=weightDict, k=k)[0]
    return tempSum / sampleNum
