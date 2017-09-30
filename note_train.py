import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import data_util
import msvcrt
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
print('num of classes: %d' % output_classes)

train_set = train_set.astype('float32')
test_set = test_set.astype('float32')
train_set /= 255
test_set /= 255
print(train_set.shape)
<<<<<<< HEAD


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

    while acc < 0.95 and iteration < 100:
        print('start iteration %d' % (iteration + 1))
        train_res = network.fit(train_set, keras.utils.to_categorical(train_labels, num_classes=output_classes),
                    epochs=10, batch_size=50,
                    validation_data=(test_set, keras.utils.to_categorical(test_labels, num_classes=output_classes)))
        train_res = train_res.history

        acc = train_res['val_acc']
        accuracies += acc if type(acc) is list else [acc]
        nets.append(NN_util.network_data(network))
        iteration += 1
        acc = acc[-1] if type(acc) is list else acc
        print('eval acc: ' + str(acc))

        if msvcrt.kbhit():
            cmd = msvcrt.getch()
            if cmd == b'x':
                print('are you sure you want to stop? (y/n)')
                if input() == 'y':
                    break


    with open('network_list.pickle', 'wb') as f:
        pickle.dump(nets, f)
    with open('accuracies.pickle', 'wb') as f:
        pickle.dump(accuracies, f)
    du.plot(accuracies)


if __name__ == '__main__':
    train_NN(reload=True)
=======
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
>>>>>>> 005085ec30d0de5b418eafd53da4443a2be9df1c
