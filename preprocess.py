import numpy as np
from sklearn.preprocessing import *
from sklearn.decomposition import PCA
from sklearn.utils import shuffle


def preprocess(dataBase, dataName, mode='min-max'):
    """

    :param dataBase:
    :param dataName:
    :param mode: min-max: (x-min)/(max-min) OR z-score: (x-mean)/std
    :return: preprocessed data

    """
    with open(dataBase + dataName, 'r') as f:
        data = f.read().strip('\n').split('\n')
        data = np.array(list(map(lambda x: list(map(eval, x.split(','))), data)))

    if mode == 'min-max':
        normP = MinMaxScaler()
    elif mode == 'z-score':
        normP = scale()
    else:
        print('Error type')
        return None, None, None, None

    pcaP = PCA()
    labelOri = data[:, -1].copy()
    data = data[:, :-1]

    sampleNum, featureNum = data.shape
    temp = np.sort(data, axis=0)
    thirdFirstValue = temp[int(sampleNum / 3)]
    thirdSecondValue = temp[int(sampleNum * 2 / 3)]

    ori = normP.fit_transform(data)
    pca = normP.fit_transform(pcaP.fit_transform(data))
    dis = data.copy().astype('float')

    for i in range(sampleNum):
        for j in range(featureNum):
            if dis[i][j] < thirdFirstValue[j]:
                dis[i][j] = 0.
            elif dis[i][j] > thirdSecondValue[j]:
                dis[i][j] = 1.
            else:
                dis[i][j] = 0.5

    ori, label = shuffle(ori, labelOri)
    pca, label = shuffle(pca, labelOri)
    dis, label = shuffle(dis, labelOri)

    return ori, pca, dis, label
