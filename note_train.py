import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import os.path
import cv2
import matplotlib.pyplot as ploter
try:
    import _pickle as pickle
except ImportError:
    import cPickle as pickle

import NN_util

input_size = 30*50
output_classes = 100
datapath = 'C:/Users/ykane/Documents/music symbols datasets/notes'


def name2label(name):
    parts = name.split('-')
    parts = parts[:3] if parts[0] == 'note' else parts[:2]
    label = '-'.join(parts)
    return label

def read_dataset():
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

    for i, name in enumerate(train_labels):
        train_labels[i] = name2label(name)
    for i, name in enumerate(test_labels):
        test_labels[i] = name2label(name)

    train_classes = sorted(np.unique(train_labels))
    test_classes = sorted(np.unique(test_labels))
    if not train_classes == test_classes:
        print('the classes are different!! danger!!')
        print('train classes: %d' % len(train_classes))
        print('test classes: %d' % len(test_classes))
        print('diff: ')
        diff = set(train_classes) - set(test_classes)
        print(diff)
        for x in diff:
            print(str(x) + ' occ in train classes: ' + str(train_classes.count(x)) + ' times')
    for i, name in enumerate(train_labels):
        train_labels[i] = train_classes.index(name)
    for i, name in enumerate(test_labels):
        test_labels[i] = test_classes.index(name)

    return [[np.array(train_set), train_labels], [np.array(test_set), test_labels]]


def plot(data):
    ploter.plot(data)
    ploter.show()


train, test = read_dataset()


output_classes = len(np.unique(train[1]))
print('type of output size: ' + str(type(output_classes)))
print(output_classes)

train_set = []
test_set = []
for x in train[0]:
    train_set.append(np.reshape(x, 30*50))
for x in test[0]:
    test_set.append(np.reshape(x, 30*50))
train_set = np.array(train_set)
test_set = np.array(test_set)
train_set = train_set.astype('float32')
test_set = test_set.astype('float32')
train_set /= 255
test_set /= 255
print(train_set.shape)
train_labels = train[1]
test_labels = test[1]

'''
permutation = np.arange(len(train_set))
np.random.shuffle(permutation)
train_set = train_set[permutation]
print(type(permutation))
print(permutation)
train_labels = train_labels[permutation]
'''

network = NN_util.create_net(input_size, output_classes)

acc = 0.0
iteration = 0
accuracies = []
nets = []
while acc < 0.9 and iteration < 150:
    print('start iteration %d' % iteration)
    train_res = network.fit(train_set, keras.utils.to_categorical(train_labels, num_classes=output_classes),
                epochs=1, batch_size=50,
                validation_data=(test_set, keras.utils.to_categorical(test_labels, num_classes=output_classes)))
    train_res = train_res.history

#    eval_res = network.evaluate(test_set, keras.utils.to_categorical(test_labels, num_classes=output_classes),
#                                batch_size=50)
    acc = train_res['val_acc']
    if type(acc) is list:
        if len(acc) > 1:
            print('len(acc) is more than 1, %d' % len(acc))
        acc = acc[0]
    accuracies.append(acc)
    nets.append(NN_util.network_data(network))
    iteration += 1
    print('eval acc: ' + str(acc))

with open('network_list', 'wb') as f:
    pickle.dump(nets, f)
with open('accuracies', 'wb') as f:
    pickle.dump(accuracies, f)
plot(accuracies)

