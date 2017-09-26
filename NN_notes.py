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
import data_util


def plot(data):
    ploter.plot(data)
    ploter.show()


class NN:

    def __init__(self):
        with open('network_list.pickle', 'rb') as f:
            nets = pickle.load(f)
        if len(nets) == 0:
            raise RuntimeError('networks not found!')

        self.network = NN_util.net_from_data(nets[-1][0], nets[-1][1])

    def classify(self, images):
        '''
        :param images: numpy array of shape (n, 1500). where n is the num of images to classify
                and 1500 is a size of each image
        :return: list of n classes for each image
        '''
        res = self.network.predict(images, batch_size=1)
        maxes = []
        for s in res:
            maxi = s.argmax()
            maxes.append(maxi)
            print([maxi, s[maxi]])
        return maxes


if __name__ == '__main__':
    if False:
        with open('accuracies', 'rb') as f:
            accuracies = pickle.load(f)
        plot(accuracies)
        exit(0)


    my_net = NN()
    du = data_util.DataUtil()

    ims = []
    ims.append(cv2.imread(("C:\\Users\\ykane\\Documents\\music symbols datasets\\notes\\test\\note-eighth-c2-2751-trans-rot3.png"), 0))
    ims.append(cv2.imread("C:\\Users\\ykane\\Documents\\music symbols datasets\\notes\\test\\note-eighth-a1-395.png", 0))
    ims.append(cv2.imread("C:\\Users\\ykane\\Documents\\music symbols datasets\\notes\\validation\\note-eighth-a1-2065-trans-rot2.png", 0))
    ims.append(cv2.imread("C:\\Users\\ykane\\Documents\\GitHub\\Notes-Recog\\cropped_notes\\note6.jpg", 0))
    ims.append(cv2.imread("C:\\Users\\ykane\\Documents\\GitHub\\Notes-Recog\\cropped_notes\\note0.jpg", 0))
    ims.append(cv2.imread("C:\\Users\\ykane\\Documents\\GitHub\\Notes-Recog\\cropped_notes\\note1.jpg", 0))

    for i, im in enumerate(ims):
        ims[i] = cv2.resize(im, (30, 50))
        ims[i] = np.reshape(im, 50*30)
    print(np.array(ims).shape)
    predicts = my_net.classify(np.array(ims))
    for p in predicts:
        print([p, du.label2name(p)])

