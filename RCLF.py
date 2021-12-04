import RWFLA
import WFLA
import frs
import numpy as np


class lowApproximationClf:
    def __init__(self):
        self.data = None
        self.label = None
        self.weight = None
        self.lLen = None

    def fit(self, data, label, choice):
        self.data = data
        self.label = label
        # Initialize weights
        weight = {i: 1 for i in range(len(data))}
        if self.lLen:
            # lLen!=0 indicates that the clf has been trained
            for i in range(self.lLen):
                weight[i] = self.weight[i]
        if choice == 2:
            # WFLA
            weight, badPoints = WFLA.gradientDescend(
                self.data, self.label, weight=weight)
        elif choice == -1:
            # FLA
            pass
        else:
            # RWFLA
            weight, badPoints = RWFLA.gradientDescend(
                self.data, self.label, weight=weight)
        
        self.weight = weight
        if self.lLen is None:
            # the first time train the clf
            self.lLen = len(self.weight)

    def predict(self, data, uniqueLabel, choice, weight=None):
        """
        Choice==3 indicates RWFLA
        Choice==2 indicates WFLA
        Choice==-1 indicates FLA
        """
        if weight is not None:
            self.weight = weight
        data = np.array(data)
        if len(data) == 1:
            data = data.reshape((1, -1))
        predictLabel = []
        for i, perData in enumerate(data):
            predictLabel.append(
                uniqueLabel[np.argmax(frs.getPositiveRegion(perData, self.data, self.label,
                                                                           weightDict=self.weight,
                                                                           k=3 if choice == 3  else 1 if choice != -1 else 0,
                                                                           uniqueLabel=uniqueLabel)[1])])
        return predictLabel

    def softVotePredict(self, data, uniqueLabel, choice):
        """
        Return lowerApproximation through soft voting
        :param data:
        :param uniqueLabel:
        :param choice:
        :return: 
        """
        data = np.array(data)
        if len(data.shape) == 1:
            data = data.reshape((-1, 1))
        lowerApprox = np.array([frs.getPositiveRegion(perData, self.data, self.label,
                                                                     weightDict=self.weight,
                                                                     k=3 if choice == 3  else 1 if choice != -1 else 0,
                                                                     uniqueLabel=uniqueLabel)[1] for perData in data])
        return lowerApprox

