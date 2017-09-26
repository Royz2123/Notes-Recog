import keras
from keras.models import Sequential
from keras.layers import Dense
try:
    import _pickle as pickle
except ImportError:
    import cPickle as pickle


def create_net(input_size, output_size):
    network = Sequential()
    network.add(Dense(units=64, activation='relu', input_dim=input_size))
    network.add(Dense(units=32, activation='sigmoid'))
    network.add(Dense(units=output_size, activation='softmax'))

    network.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return network


def network_data(net):
    json = net.to_json()
    weights = net.get_weights()
    return json, weights


def net_from_data(json, weights):
    net = keras.models.model_from_json(json)
    net.set_weights(weights)
    net.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return net


def save_networks(nets):
    to_save = []
    for net in nets:
        to_save.append(network_data(net))
    with open('net_list', 'wb') as f:
        pickle.dump(f, to_save)
