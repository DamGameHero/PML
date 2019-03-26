import sys
import numpy as np
import pandas as pd
import inspect
import re


def describe(arg):
    frame = inspect.currentframe()
    callerframeinfo = inspect.getframeinfo(frame.f_back)
    try:
        context = inspect.getframeinfo(frame.f_back).code_context
        caller_lines = ''.join([line.strip() for line in context])
        m = re.search(r'describe\s*\((.+?)\)$', caller_lines)
        if m:
            caller_lines = m.group(1)
            position = str(callerframeinfo.filename) + "@" + str(callerframeinfo.lineno)

            # Add additional info such as array shape or string length
            additional = ''
            if hasattr(arg, "shape"):
                additional += "[shape={}]".format(arg.shape)
            elif hasattr(arg, "__len__"):  # shape includes length information
                additional += "[len={}]".format(len(arg))

            # Use str() representation if it is printable
            str_arg = str(arg)
            str_arg = str_arg if str_arg.isprintable() else repr(arg)

            print(position, "describe(" + caller_lines + ") = ", end='')
            print(arg.__class__.__name__ + "(" + str_arg + ")", additional)
        else:
            print("Describe: couldn't find caller context")

    finally:
        del frame
        del callerframeinfo


def get_data(args):
    path = "data.csv"
    if len(args) < 2:
        print("No arguments given. Try with \"data.csv\".")
    else:
        path = args[1]
    try:
        data = pd.read_csv(path, header=None)
    except Exception as e:
        print("Can't extract data from {}.".format(path))
        print(e.__doc__)
        sys.exit(0)
    return data


class layer:
    def __init__(self, size, activation='sigmoid'):
        self.size = size
        self.activation = activation


class network:
    def __init__(self, layers, data_train, data_valid):
        self.layers = layers
        self.size = len(layers)
        self.train_size = len(data_train)
        self.valid_size = len(data_valid)
        self.thetas = []
        self.deltas = []
        i = 0
        while i < self.size - 1:
            self.thetas.append(theta_init(
                self.layers[i].size, self.layers[i + 1].size))
            self.deltas.append(theta_init(
                self.layers[i].size, self.layers[i + 1].size, 0.0))
            i += 1


def gradient_descent(network, loss='cross_entropy', learning_rate=1.0, turns=100):
    i = 0
    j = 0
    while i < turns:
        derivate = backward_pro(network)
        while j < network.size - 1:
            network.thetas[j] = network.thetas[j] - learning_rate * derivate[j]
    return 1


def theta_init(layer_1, layer_2, eps=0.5):
    return np.random.rand(layer_2, layer_1 + 1) * 2 * eps - eps


def forward_pro(network, row):
    activ_dict = {
            'sigmoid': sigmoid,
    }
    a = [row]
    i = 0
    while i < network.size:
        a.append(activ_dict[network.layers[i].activation](
            a[i] * network.thetas[i]))
        i += 1
    return a


def backward_pro(network):
    i = 0
    delta = [0] * (network.size)
    total_delta = network.deltas.copy()
    derivate = [0] * (network.size - 1)
    while i < len(network.x):
        a = forward_pro(network, network.x[i])
        j = network.size - 1
        delta[j] = a[j] - network.y[i]
        j -= 1
        while j > 0:
            delta[j] = network.thetas[j] * delta[j + 1] * a[j] * (1 - a[j])
            total_delta[j] += delta[j + 1] * np.transpose(a[j])
            j -= 1
        total_delta[j] += delta[j + 1] * np.transpose(a[j])
        i += 1
    i = 0
    while i < network.size:
        derivate[i] = total_delta[i] / network.train_size
        i += 1
    return derivate


def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))


def softmax(h):
    return np.exp(-1 * h) / (np.sum(np.exp(-1 * h)))


def cross_entropy(predict, y_class):
    size = np.size(predict, 0)
    return ((1 / size)
            * (-1 * y_class.dot(np.transpose(np.log(predict)))
            - (1 - y_class).dot(np.transpose(np.log(1 - predict)))))


def main():
    pd.set_option('display.expand_frame_repr', False)
    df = get_data(sys.argv)
    pd.set_option('display.max_rows', len(df))
    df = df.rename(columns={0: "id", 1: "Class"})
    df = df.drop(columns='id')
    df = df.sample(frac=1)
    dfs = np.split(df, [int((len(df) * 0.80))], axis=0)
    describe(dfs[0])
    describe(dfs[1])
    # describe(forward_pro(2, 3))
    # print(df.describe())
    # print(df)


if __name__ == '__main__':
    main()
