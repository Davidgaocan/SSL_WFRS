import numpy as np
import RCLF

uniqueLabel = []


def subSample(L, num):
    """

    Delete num samples from L 
    :param L:list, index of selected samples
    :param num: number of samples to be deleted
    :return:
    """
    for _ in range(int(num)):
        L.remove(np.random.choice(L))
    return L


def measureError(clf1, p1, clf2, p2, label):
    """
    Measure current error of clf1 and clf2 on labeled data
    :param label:
    :param clf1:
    :param p1:corresponding modal-data for clf1
    :param clf2:
    :param p2:corresponding modal-data for clf2
    :return:
    """

    pre1 = np.array(clf1.predict(p1, uniqueLabel, choice=3))
    pre2 = np.array(clf2.predict(p2, uniqueLabel, choice=3))
    pre = pre1[pre1 == pre2]
    tempLabel = label[pre1 == pre2]
    if not np.sum(pre1 == pre2):
        return 0.5
    return np.sum(pre != tempLabel) / np.sum(pre1 == pre2)


def LabelU(clf, data, U):
    """
    Give pseudo-label to U
    :param clf:sets of clfs
    :param data: 
    :param U: index of unlabeled data
    :return: the pseudo-label of U
    """

    label = uniqueLabel[np.argmax(0.4 * clf[0].softVotePredict(data[0][U[0]], uniqueLabel=uniqueLabel, choice=3) +
                                  0.4 * clf[1].softVotePredict(data[1][U[1]], uniqueLabel=uniqueLabel, choice=3) +
                                  0.2 * clf[2].softVotePredict(data[2][U[2]], uniqueLabel=uniqueLabel, choice=3), axis=1)]
    # soft voting
    return label


def triTraining(data1, data2, data3, labelOri, trainIdx, splitRatio=0.1):
    global uniqueLabel
    """
    
    :param splitRatio: 
    :param trainIdx:
    :param data1: ori
    :param data2: pca
    :param data3: dis
    :param label: 
    :return:
    """
    data = [data1, data2, data3]
    label = labelOri.copy()
    uniqueLabel = np.unique(label)
    clf = [RCLF.lowApproximationClf(), RCLF.lowApproximationClf(),
           RCLF.lowApproximationClf()]

    # RWFLA
    clfClfInfo = [[0.5], [0.5], [0.5]]
    trainNum = len(trainIdx)
    L = []
    U = []
    L_ = [[], [], []]
    e__ = [0.5, 0.5, 0.5]
    l__ = [0, 0, 0]
    e_ = [0, 0, 0]
    update = [False, False, False]
    border = int(np.ceil(trainNum * splitRatio))
    # the border between labeled data and unlabled data

    for i in range(3):
        L.append(trainIdx[:border].tolist())
        U.append(trainIdx[border:])
        clf[i].fit(data[i][L[i]], label[L[i]], choice=3)
        # using RWFLA classifiers as base classifiers

    while True:
        # Tri-Training
        for i in range(3):
            L_[i] = []
            update[i] = False
            e_[i] = measureError(clf[(i + 1) % 3], data[(i + 1) % 3][trainIdx[:border]], clf[(i + 2) % 3],
                                 data[(i + 2) % 3][trainIdx[:border]], label[trainIdx[:border]].copy())
            clfClfInfo[i].append(e_[i])
            if e_[i] < e__[i]:
                # If one clf achieves better results on L compared to the last time
                for possibleIdx in U[0]:
                    # Choose samples from U for clf_i to learn in this iteration
                    if clf[(i + 1) % 3].predict(data[(i + 1) % 3][possibleIdx].reshape(1, -1), uniqueLabel, choice=3) \
                            == clf[(i + 2) % 3].predict(data[(i + 2) % 3][possibleIdx].reshape(1, -1), uniqueLabel, 3):
                        L_[i].append(possibleIdx)
                if not l__[i]:
                    l__[i] = np.floor(e_[i] / (e__[i] - e_[i]) + 1)
                if l__[i] < len(L_[i]):
                    # Not all samples chosen need to be learned
                    if e_[i] * len(L_[i]) < e__[i] * l__[i]:
                        update[i] = True
                    elif l__[i] > e_[i] / (e__[i] - e_[i]):
                        L_[i] = subSample(L_[i], np.ceil(
                            e__[i] * l__[i] / e_[i] - 1))
                        update[i] = True
        for i in range(3):
            if update[i]:
                # Update specific clfs that satisfiy above conditions
                tempL = L[i].copy()
                tempL.extend(L_[i])
                tempLabel = label[tempL]
                for perI in L_[i]:
                    tempLabel[tempL.index(perI)] = clf[(i + 1) % 3].predict(data[(i + 1) % 3][perI].reshape(1, -1),
                                                                            uniqueLabel, choice=3)[0]

                clf[i].fit(data[i][tempL], tempLabel, choice=3)
                e__[i] = e_[i]
                l__[i] = len(L_[i])

        if not any(update):
            break

    labelU = LabelU(clf, data, U)
    # Label unlabeled data
    label[trainIdx[border:]] = labelU
    # Replace original label with pseudo-label
    return label, border, trainIdx, clf
