import numpy as np
import RWFLA


def getDistance(x, y):
    """
    Return Euclidean distance
    """
    x, y = np.array(x), np.array(y)
    return np.linalg.norm(x - y)


def getLoss(data, nearestSamples, weight, sita):
    """
    Return loss under current samples' weights, similar to RWFLA
    """
    sampleNum = data.shape[0]
    tempSum = 0
    for i in range(sampleNum):
        tempSum += 1 - np.exp(
            -weight[nearestSamples[i][1][0]] ** 2 * getDistance(data[i], data[nearestSamples[i][1][0]]) ** 2 / sita)
    return tempSum / sampleNum


def gradientDescend(data, label, weight=None):
    sampleNum, featureNum = data.shape
    if weight is not None:
        weight = weight
    else:
        weight = {i: 1 for i in range(sampleNum)}
    nearestSamples = {i: RWFLA.getKHandM(
        i, data, label, k=1) for i in range(sampleNum)}
    
    kernelParameter = 0.15
    l1 = 0
    l2 = 1
    lr = 0.1
    thresh = 0.001
    while True:
        if abs(l1 - l2) < thresh:
            break
        l1 = l2
        for i in range(sampleNum):
            weight[nearestSamples[i][1][0]] += lr * 2 / kernelParameter * getDistance(data[i], data[nearestSamples[i][1][0]]) ** 2 * weight[
                nearestSamples[i][1][0]] * np.exp(-(weight[nearestSamples[i][1][0]] * getDistance(data[i], data[nearestSamples[i][1][0]])) ** 2 / kernelParameter)
        l2 = getLoss(data, nearestSamples, weight, kernelParameter)

    badPoints = []
    return weight, badPoints
