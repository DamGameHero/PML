import sys
import numpy as np
import pandas as pd
import inspect
import re
import math
import matplotlib.pyplot as plt
from scipy import stats as astats
import copy


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
    seed_id = 2019

    def __init__(self, size, activation='sigmoid'):
        self.size = size
        self.activation = activation
        self.seed = layer.seed_id
        layer.seed_id += 1


class network:
    def __init__(self, layers, data_train, data_valid):
        self.layers = layers
        self.size = len(layers)
        self.train_size = len(data_train)
        self.valid_size = len(data_valid)
        self.x = data_train.drop(columns=['class', 'vec_class']).to_numpy()
        self.valid_x = data_valid.drop(columns=['class', 'vec_class']).to_numpy()
        self.y = data_train['class'].to_numpy()
        self.vec_y = np.asarray(data_train['vec_class'].tolist())
        self.valid_y = data_valid['class'].to_numpy()
        self.valid_vec_y = np.asarray(data_valid['vec_class'].tolist())
        self.thetas = []
        self.deltas = []
        self.predict = []
        self.valid_predict = []
        i = 0
        while i < self.size - 1:
            self.thetas.append(theta_init(
                self.layers[i].size,
                self.layers[i + 1].size,
                self.layers[i].seed))
            self.deltas.append(theta_init(
                self.layers[i].size, self.layers[i + 1].size, eps=0.0))
            i += 1


def gradient_descent(network, loss='cross_entropy', learning_rate=1.0, turns=80):
    costs = []
    valid_costs = []
    # new_cost = cost(theta, x, y_class)[0]
    # costs.append(new_cost)
    i = 0
    while i < turns:
        derivate = backward_pro(network)
        j = 0
        while j < network.size - 1:
            network.thetas[j] = network.thetas[j] - learning_rate * derivate[j]
            j += 1
        new_cost = cross_entropy(np.asarray(network.predict), network.vec_y)
        new_valid_cost = cross_entropy(
                np.asarray(network.valid_predict), network.valid_vec_y)
        costs.append(new_cost)
        valid_costs.append(new_valid_cost)
        network.predict.clear()
        network.valid_predict.clear()
        i += 1
    print("train cost = ", new_cost)
    print("valid cost = ", new_valid_cost)
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost Function')
    plt.title("Cost Function Evolution")
    plt.plot(
            np.arange(turns),
            costs)
    plt.plot(
            np.arange(turns),
            valid_costs)
    plt.show()
    return 1


def theta_init(layer_1, layer_2, seed=0, eps=0.5):
    np.random.seed(seed)
    return np.random.rand(layer_2, layer_1 + 1) * 2 * eps - eps


def forward_pro(network, row, predict=True):
    activ_dict = {
            'sigmoid': sigmoid,
    }
    a = [row.reshape(-1, 1)]
    i = 0
    while i < network.size - 1:
        a[i] = np.insert(a[i], 0, 1.0, axis=0)
        a.append(activ_dict[network.layers[i].activation](
            network.thetas[i].dot(a[i])))
        i += 1
    if predict:
        network.predict.append(a[i])
    else:
        network.valid_predict.append(a[i])

    return a


def backward_pro(network):
    i = 0
    delta = [0] * (network.size)
    total_delta = copy.deepcopy(network.deltas)
    derivate = [0] * (network.size - 1)
    while i < len(network.x):
        if i < len(network.valid_x):
            forward_pro(network, network.valid_x[i], predict=False)
        a = forward_pro(network, network.x[i])
        j = network.size - 1
        delta[j] = a[j] - network.vec_y[i].reshape(-1, 1)
        j -= 1
        while j > 0:
            delta[j] = np.transpose(network.thetas[j]).dot(delta[j + 1]) * a[j] * (1 - a[j])
            total_delta[j] += delta[j + 1] * np.transpose(a[j])
            delta[j] = delta[j][1:, :]
            j -= 1
        total_delta[j] += delta[j + 1] * np.transpose(a[j])
        i += 1
    i = 0
    while i < network.size - 1:
        derivate[i] = total_delta[i] / network.train_size
        i += 1
    return derivate


def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))


def softmax(h):
    results = []
    i = 0
    while i < len(h):
        results.append(np.exp(-1 * h[i]) / (np.sum(np.exp(-1 * h))))
        i += 1
    return results


def cross_entropy(predict, y_class):
    size = np.size(predict, 0)
    predict = predict.reshape(-1, 2)
    # y_0 = (-1 * y_class[:, 0].T.dot((np.log(predict[:, 0]))) - (1 - y_class[:, 0]).T.dot((np.log(1 - predict[:, 0]))))
    # y_1 = (-1 * y_class[:, 1].T.dot((np.log(predict[:, 1]))) - (1 - y_class[:, 1]).T.dot((np.log(1 - predict[:, 1]))))
    # return (1 / size) * (y_0 + y_1)
    return ((1 / size)
            * (-1 * y_class[:, 0].dot((np.log(predict[:, 0])))
                - (1 - y_class[:, 0]).dot((np.log(1 - predict[:, 0])))))


def get_stats(df):
    # df = df.select_dtypes(include='number') # a faire
    # df = df[(np.abs(astats.zscore(df)) < 3).all(axis=1)] # a faire
    stats = {
            column: escribe(sub_dict)
            for column, sub_dict in df.select_dtypes(include='number').to_dict().items()}
    return stats


def percentile(percent, count, values):
    x = percent * (count - 1)
    return (values[math.floor(x)]
            + (values[math.floor(x) + 1] - values[math.floor(x)]) * (x % 1))


def escribe(data):
    clean_data = {k: data[k] for k in data if not np.isnan(data[k])}
    values = np.sort(np.array(list(clean_data.values()), dtype=object))
    count = len(clean_data)
    stats = {'count': count}
    stats['mean'] = sum(clean_data.values()) / count
    stats['var'] = (
            1
            / (count - 1)
            * np.sum(np.power(values - stats['mean'], 2)))
    stats['std'] = np.sqrt(stats['var'])
    # stats['min'] = values[0]
    # stats['max'] = values[count - 1]
    # stats['range'] = stats['max'] - stats['min']
    # stats['25%'] = percentile(0.25, count, values)
    # stats['75%'] = percentile(0.75, count, values)
    # if count % 2 == 0:
    #     stats['50%'] = (values[int(count / 2 - 1)]
    #                     + values[int(count / 2)]) / 2
    # else:
    #     stats['50%'] = values[int((count + 1) / 2 - 1)]
    # stats['Q3-Q1 range'] = stats['75%'] - stats['25%']
    # stats['mad'] = np.sum(np.absolute(values - stats['mean'])) / count
    # stats['10%'] = percentile(0.1, count, values)
    # stats['20%'] = percentile(0.2, count, values)
    # stats['30%'] = percentile(0.3, count, values)
    # stats['40%'] = percentile(0.4, count, values)
    # stats['60%'] = percentile(0.6, count, values)
    # stats['70%'] = percentile(0.7, count, values)
    # stats['80%'] = percentile(0.8, count, values)
    # stats['90%'] = percentile(0.9, count, values)
    # svalues = [
    #         item for item in clean_data.values()
    #         if item >= stats['10%'] and item <= stats['90%']]
    # stats['clmean'] = sum(svalues) / len(svalues)
    return stats


def display_stats(stats):
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.float_format', lambda x: ' %.6f' % x)
    res = pd.DataFrame.from_dict(stats)
    res = res.reindex([
        'count', 'mean', 'std', 'min',
        '25%', '50%', '75%', 'max'])
    print(res)


def feature_scaling(df, stats):
    for subj in stats:
        df[subj] = (df[subj] - stats[subj]['mean']) / stats[subj]['std']
    return df


def main():
    df = get_data(sys.argv)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_rows', len(df))
    df = df.rename(columns={0: "id", 1: "class"})
    df = df.drop(columns=['id'])
    #df = df.drop(columns=['id', 20, 13, 11, 16])
    #df = df.drop(columns=['id', 4, 5, 24, 25, 20, 13, 11, 16, 14, 15, 29, 9, 22])
    #df = df.drop(columns=['id', 4, 5, 24, 25, 20, 13, 11, 16, 14, 15, 29, 9, 22, 26, 21, 19, 15, 10, 6])
    stats = get_stats(df)
    df = feature_scaling(df, stats)
    df['class'] = df['class'].map({'M': 1, 'B': 0})
    df['vec_class'] = df['class'].map({1: [1, 0], 0: [0, 1]})
    #df = df.sample(frac=1)
    dfs = np.split(df, [int((len(df) * 0.80))], axis=0)
    layers = [
            layer(len(df.columns) - 2),
            layer(24),
            layer(24),
            layer(2)]
    net = network(layers, dfs[0], dfs[1])
    gradient_descent(net)


if __name__ == '__main__':
    main()
