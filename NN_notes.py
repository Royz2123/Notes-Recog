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


class NN:

    def __init__(self):
        with open('network_list', 'rb') as f:
            nets = pickle.load(f)
        if len(nets) == 0:
            raise ValueError('networks not found!')

        self.network = NN_util.net_from_data(nets[-1][0], nets[-1][1])

    def classify(self, image):
        res = self.network.predict(image, batch_size=1)
        print(res)
        return res


if __name__ == '__main__':
    if False:
        with open('accuracies', 'rb') as f:
            accuracies = pickle.load(f)
        plot(accuracies)
        exit(0)


    my_net = NN()

    im = cv2.imread(("C:\\Users\\ykane\\Documents\\music symbols datasets\\notes\\test\\note-eighth-c2-2751-trans-rot3.png"), 0)
    print(im.shape)
    im = cv2.resize(im, (30, 50))
    # cv2.imshow('1', im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    im = np.reshape(im, 50*30)
    im = im.tolist()
    my_net.classify(np.array([im]))

