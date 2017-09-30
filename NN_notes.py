import keras
import numpy as np
import os.path
import matplotlib.pyplot as ploter
import cv2
try:
    import _pickle as pickle
except ImportError:
    import cPickle as pickle

import NN_util


def plot(data):
    ploter.plot(data)
    ploter.show()


class NN(object):
    def __init__(self):
        with open('network_list', 'rb') as f:
            nets = pickle.load(f)
        if len(nets) == 0:
            raise ValueError('networks not found!')

        self.network = NN_util.net_from_data(nets[-1][0], nets[-1][1])

    def classify(self, image):
        return self.network.predict(image, batch_size=1)
