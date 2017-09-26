import numpy as np
import os.path
import cv2
import matplotlib.pyplot as ploter
try:
    import _pickle as pickle
except ImportError:
    import cPickle as pickle


class DataUtil:

    def __init__(self):
        if os.path.exists('name_list.pickle'):
            print('name_list exists')
            with open('name_list.pickle', 'rb') as f:
                self.name_list = pickle.load(f)
        else:
            self.name_list = None

    def names2label(self, names):
        for i, name in enumerate(names):
            parts = name.split('-')
            parts = parts[:3] if parts[0] == 'note' else parts[:2]
            label = '-'.join(parts)
            names[i] = label

        classes = sorted(np.unique(names))
        self.name_list = classes.copy()
        with open('name_list.pickle', 'wb') as f:
            pickle.dump(self.name_list, f)

        for i, l in enumerate(names):
            names[i] = classes.index(l)

        return classes

    def label2name(self, label):
        try:
            return self.name_list[label]
        except IndexError or TypeError:
            return ''

    def read_dataset(self, datapath):
        print('reading dataset')
        if not os.path.exists(os.path.join(datapath, 'train.pickle')) or\
                not os.path.exists(os.path.join(datapath, 'test.pickle')):
            train_labels = [name for name in os.listdir(os.path.join(datapath, 'train'))
                            if os.path.splitext(name)[1] == '.png']
            train_set = [cv2.imread(os.path.join(os.path.join(datapath, 'train'), f), 0) for f in train_labels]
            test_labels = [name for name in os.listdir(os.path.join(datapath, 'test'))
                           if os.path.splitext(name)[1] == '.png']
            test_set = [cv2.imread(os.path.join(os.path.join(datapath, 'test'), f), 0) for f in test_labels]
            with open(os.path.join(datapath, 'train.pickle'), 'wb') as f:
                pickle.dump([train_set, train_labels], f)
            with open(os.path.join(datapath, 'test.pickle'), 'wb') as f:
                pickle.dump([test_set, test_labels], f)
        else:
            with open(os.path.join(datapath, 'train.pickle'), 'rb') as f:
                train_set, train_labels = pickle.load(f)
            with open(os.path.join(datapath, 'test.pickle'), 'rb') as f:
                test_set, test_labels = pickle.load(f)

        train_classes = self.names2label(train_labels)
        test_classes = self.names2label(test_labels)

        if not train_classes == test_classes:
            print('the classes are different!! danger!!')
            print('train classes: %d' % len(train_classes))
            print('test classes: %d' % len(test_classes))
            print('diff: ')
            diff = set(train_classes) ^ set(test_classes)
            print(diff)
            for x in diff:
                print(str(x) + ' occ in train classes: ' + str(train_classes.count(x)) + ' times')
                print(str(x) + ' occ in test classes: ' + str(test_classes.count(x)) + ' times')

        train_set_reshape = []
        test_set_reshape = []
        for x in train_set:
            train_set_reshape.append(np.reshape(x, 30 * 50))
        for x in test_set:
            test_set_reshape.append(np.reshape(x, 30 * 50))

        return [[np.array(train_set_reshape), np.array(train_labels)], [np.array(test_set_reshape), np.array(test_labels)]]

    @staticmethod
    def plot(data):
        ploter.plot(data)
        ploter.show()

