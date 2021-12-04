import numpy as np
from WFLA import getDistance


def getNearestKSample(sampleIdx, data, k):
    """

    :param sampleIdx: 
    :param data:
    :param k: k nearest samples
    :return: samples' idx
    """
    distance = [getDistance(data[sampleIdx], data[i])
                for i in range(len(data))]
    argDistance = np.argsort(distance)
    return argDistance[1:k + 1]


def getKHandM(sampleIdx, data, label, k=1, badPoints=None):
    """
    For sample Idx, obtain k nearest heterogeneous and homogeneous samples, which are not in badPoints
    :param sampleIdx: idx
    :param data:
    :param label:
    :param k:
    :param badPoints:
    :return:
    """
    if badPoints is None:
        badPoints = []
    distance = [getDistance(data[sampleIdx], data[i])
                for i in range(len(data))]
    argDistance = np.argsort(distance)
    nearestHit = []
    nearestMiss = []
    for i in argDistance:
        if len(nearestHit) == k and len(nearestMiss) == k:
            return nearestHit, nearestMiss
        if i == sampleIdx:
            continue
        if label[i] == label[sampleIdx] and len(nearestHit) < k and i not in badPoints:
            nearestHit.append(i)
            continue
        if label[i] != label[sampleIdx] and len(nearestMiss) < k and i not in badPoints:
            nearestMiss.append(i)
            continue
    return nearestHit, nearestMiss


def getBadPoints(data, label, k):
    """
    :param data:
    :param label:
    :param k: similar to the k parameter in KNN
    :return:
    """
    badPoints = []
    label = np.array(label)
    for i in range(len(data)):
        sameLabelNum = list(
            label[getNearestKSample(i, data, k)]).count(label[i])
        if sameLabelNum <= k * 0.25:
            badPoints.append(i)
    return badPoints


def getDistanceWithKN(data, sampleIdx, nearestSamples):
    """
    Get average distance
    :param data:
    :param sampleIdx:
    :param nearestSamples: the nearest samples of sample idx
    :return:
    """
    tempSum = 0
    nearestSamples = np.array(nearestSamples).reshape((1, -1))
    for i in nearestSamples:
        tempSum += getDistance(data[i], data[sampleIdx])
    return tempSum / len(nearestSamples)


def getImprovedLoss(data, weight, nearestSamples, badPoints=None, kernelP=0.15):
    """
    Loss under current samples' weights 
    :param kernelP:
    :param k:
    :param badPoints:
    :param data:
    :param weight:
    :param nearestSamples: 
    :return: 
    """
    if badPoints is None:
        badPoints = []
    loss = 0
    for i in range(len(data)):
        if i not in badPoints:
            for j in range(len(nearestSamples[i][1])):
                loss += 1 - np.exp(-weight[nearestSamples[i][1][j]] ** 2 * getDistance(
                    data[i], data[nearestSamples[i][1][j]]) ** 2 / kernelP)
    return loss


def gradientDescend(data, label, k=5, weight=None):
    kernelParameter = 0.15
    sampleNum, featureNum = data.shape
    if weight is not None:
        weight = weight
    else:
        weight = {i: 1 for i in range(sampleNum)}
        badPoints = getBadPoints(data, label, k=9)

    nearestSample = {i: getKHandM(i, data, label, k=k, badPoints=badPoints)
                     for i in range(sampleNum) if i not in badPoints}

    lr = 0.1
    threshold = 0.001

    loss = getImprovedLoss(data, weight, nearestSample,
                           badPoints=badPoints, k=k)
    while True:
        # updating
        for i in range(sampleNum):
            if i in badPoints:
                # No updation for badPoints
                continue
            for j in range(len(nearestSample[i][1])):
                weight[nearestSample[i][1][j]] += lr * 2 / kernelParameter * getDistance(data[i], data[nearestSample[i][1][j]]) ** 2 * weight[nearestSample[i][1][j]] * np.exp(
                    -(weight[nearestSample[i][1][j]] * getDistance(data[i], data[nearestSample[i][1][j]])) ** 2 / kernelParameter)

        temp = getImprovedLoss(
            data, weight, nearestSample, badPoints=badPoints, k=k)

        if abs(temp - loss) < threshold:
            break
        else:
            loss = temp

    for i in badPoints:
        # Give badPoints smallest weight
        weight[i] = 1e-6

    return weight, badPoints
