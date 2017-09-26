import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import data_util
try:
    import _pickle as pickle
except ImportError:
    import cPickle as pickle

import NN_util

input_size = 30*50
output_classes = 100
datapath = 'C:/Users/ykane/Documents/music symbols datasets/notes'

du = data_util.DataUtil()
[train_set, train_labels], [test_set, test_labels] = du.read_dataset(datapath)


output_classes = len(np.unique(train_labels))
print('type of output size: ' + str(type(output_classes)))
print(output_classes)

train_set = train_set.astype('float32')
test_set = test_set.astype('float32')
train_set /= 255
test_set /= 255
print(train_set.shape)


def train_NN(reload=False):
    if not reload:
        network = NN_util.create_net(input_size, output_classes)
        accuracies = []
        nets = []
        acc = 0.0
        iteration = 0
    else:
        with open('network_list.pickle', 'rb') as f:
            nets = pickle.load(f)
        if len(nets) == 0:
            raise RuntimeError('networks not found!')
        network = NN_util.net_from_data(nets[-1][0], nets[-1][1])

        with open('accuracies.pickle', 'rb') as f:
            accuracies = pickle.load(f)
        acc = 0.0 if len(accuracies) == 0 else accuracies[-1]
        iteration = len(nets)

    while acc < 0.99 and iteration < 100:
        print('start iteration %d' % iteration)
        train_res = network.fit(train_set, keras.utils.to_categorical(train_labels, num_classes=output_classes),
                    epochs=10, batch_size=50,
                    validation_data=(test_set, keras.utils.to_categorical(test_labels, num_classes=output_classes)))
        train_res = train_res.history

        acc = train_res['val_acc']
        if type(acc) is list:
            if len(acc) > 1:
                print('len(acc) is more than 1, %d' % len(acc))
        accuracies += acc
        nets.append(NN_util.network_data(network))
        iteration += 1
        acc = acc[-1]
        print('eval acc: ' + str(acc))

    with open('network_list.pickle', 'wb') as f:
        pickle.dump(nets, f)
    with open('accuracies.pickle', 'wb') as f:
        pickle.dump(accuracies, f)
    du.plot(accuracies)


if __name__ == '__main__':
    train_NN(True)
