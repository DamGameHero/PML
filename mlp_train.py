import sys
import numpy as np
import pandas as pd
import inspect
import re
import math
import matplotlib.pyplot as plt
from scipy import stats as astats
import copy
import timeit
import argparse


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


def check_fpositive(value):
    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive value" % value)
    return fvalue


def check_fpositive_null(value):
    fvalue = float(value)
    if fvalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive or null value" % value)
    return fvalue


def check_ipositive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive value" % value)
    return ivalue


def check_ipositive_null(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive or null value" % value)
    return ivalue


def check_outliers(value):
    fvalue = float(value)
    if fvalue < 0.0 or fvalue > 4.0:
        raise argparse.ArgumentTypeError("%s is an invalid z score (must be 0 < z < 3)" % value)
    return fvalue


def get_data():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="a data set", nargs='?', default="data.csv")
    parser.add_argument("-L", "--layers", help="Number of layers", type=check_ipositive, default=2)
    parser.add_argument("-U", "--units", help="Number of units per layer", type=check_ipositive, default=12)
    parser.add_argument("-lr", "--learning_rate", help="Learning Rate's value", type=check_fpositive, default=1.0)
    parser.add_argument("-b", "--batch_size", help="Size of batch", type=check_ipositive, default=0)
    parser.add_argument("-e", "--epochs", help="Number of epochs", type=check_ipositive, default=80)
    parser.add_argument("-la", "--lmbd", help="Lambda's value for regularization", type=check_fpositive_null, default=0.0)
    parser.add_argument("-o", "--outliers", help="Drop outliers with the z score given", type=check_outliers, default=0.0)
    parser.add_argument("-shu", "--shuffle", help="Shuffle the data set", action="store_true")
    parser.add_argument("-nag", "--nesterov", help="Nesterov’s Accelerated Momentum Gradient optimization", action="store_true")
    parser.add_argument("-mu", "--momentum", help="Momentum 's value fot NAG (Nesterov's Accelerated Momentum)", type=check_fpositive, default=0.01)
    parser.add_argument("-adg", "--adagrad", help="Adagrad optimization", action="store_true")
    parser.add_argument("-rms", "--rmsprop", help="RMSprop optimization", action="store_true")
    args = parser.parse_args()
    try:
        data = pd.read_csv(args.data, header=None)
    except Exception as e:
        print("Can't extract data from {}.".format(args.data))
        print(e.__doc__)
        sys.exit(0)
    return data, args


class layer:
    seed_id = 2019

    def __init__(self, size, activation='sigmoid'):
        self.size = size
        self.activation = activation
        self.seed = layer.seed_id
        layer.seed_id += 1


class network:
    def __init__(self, layers, data_train, data_valid, args):
        self.layers = layers
        self.size = len(layers)
        self.train_size = len(data_train)
        self.valid_size = len(data_valid)
        self.x = data_train.drop(columns=['class', 'vec_class']).to_numpy()
        self.batched_x = 0
        self.batched_vec_y = 0
        self.valid_x = data_valid.drop(columns=['class', 'vec_class']).to_numpy()
        self.y = data_train['class'].to_numpy()
        self.vec_y = np.asarray(data_train['vec_class'].tolist())
        self.valid_y = data_valid['class'].to_numpy()
        self.valid_vec_y = np.asarray(data_valid['vec_class'].tolist())
        self.lmbd = args.lmbd
        self.momentum = args.momentum
        self.velocity = []
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
            self.velocity.append(theta_init(
                self.layers[i].size, self.layers[i + 1].size, eps=0.0))
            i += 1

    def split(self, batch_size):
        sections = []
        index = batch_size
        while index + batch_size <= self.train_size:
            sections.append(index)
            index += batch_size
        self.batched_x = np.split(self.x, sections)
        self.batched_vec_y = np.split(self.vec_y, sections)


def add_cost(net, costs, valid_costs):
    new_cost = cross_entropy(np.asarray(net.predict), net.vec_y, net.lmbd, net)
    new_valid_cost = cross_entropy(
            np.asarray(net.valid_predict), net.valid_vec_y, 0, net)
    costs.append(new_cost)
    valid_costs.append(new_valid_cost)
    net.predict.clear()
    net.valid_predict.clear()


def plot_results(costs, valid_costs, epochs):
    plt.xlabel('No. of epochs')
    plt.ylabel('Cost Function')
    plt.title("Cost Function Evolution")
    plt.plot(
            np.arange(epochs),
            costs)
    plt.plot(
            np.arange(epochs),
            valid_costs)
    plt.show()


def display_results(costs, valid_costs, epochs):
    print(all(costs[i] >= costs[i+1] for i in range(len(costs)-1)))
    print(all(valid_costs[i] >= valid_costs[i+1] for i in range(len(valid_costs)-1)))
    print("train cost 5 = ", costs[epochs-5])
    print("valid cost 5 = ", valid_costs[epochs-5])
    print("train cost 4 = ", costs[epochs-4])
    print("valid cost 4 = ", valid_costs[epochs-4])
    print("train cost 3 = ", costs[epochs-3])
    print("valid cost 3 = ", valid_costs[epochs-3])
    print("train cost 2 = ", costs[epochs-2])
    print("valid cost 2 = ", valid_costs[epochs-2])
    print("train cost = ", costs[epochs-1])
    print("valid cost = ", valid_costs[epochs-1])
    plot_results(costs, valid_costs, epochs)


def gradient_descent(net, loss='cross_entropy', learning_rate=1.0, batch_size=0, epochs=80):
    start = timeit.default_timer()
    costs = []
    valid_costs = []
    i = 0
    while i < epochs:
        derivate = backward_pro(net)
        j = 0
        while j < net.size - 1:
            net.thetas[j] = net.thetas[j] - learning_rate * derivate[j]
            j += 1
        add_cost(net, costs, valid_costs)
        i += 1
    stop = timeit.default_timer()
    print('Time Gradient: ', stop - start)
    display_results(costs, valid_costs, epochs)
    return 1


def gradient_descent_rms(net, loss='cross_entropy', learning_rate=1.0, batch_size=0, epochs=80):
    start = timeit.default_timer()
    costs = []
    valid_costs = []
    i = 0
    decay_rate = 0.9
    eps = 0.00001
    cache = copy.deepcopy(net.deltas)
    while i < epochs:
        derivate = backward_pro(net)
        j = 0
        while j < net.size - 1:
            cache[j] = decay_rate * cache[j] + (1 - decay_rate) * derivate[j]**2
            net.thetas[j] = net.thetas[j] - learning_rate * derivate[j] / (np.sqrt(cache[j]) + eps)
            j += 1
        add_cost(net, costs, valid_costs)
        i += 1
    stop = timeit.default_timer()
    print('Time to perform RMSprop Gradient: ', stop - start)
    display_results(costs, valid_costs, epochs)
    return 1


def gradient_descent_adg(net, loss='cross_entropy', learning_rate=1.0, batch_size=0, epochs=80):
    start = timeit.default_timer()
    costs = []
    valid_costs = []
    i = 0
    eps = 0.00001
    cache = copy.deepcopy(net.deltas)
    while i < epochs:
        derivate = backward_pro(net)
        j = 0
        while j < net.size - 1:
            cache[j] += derivate[j]**2
            net.thetas[j] = net.thetas[j] - learning_rate * derivate[j] / (np.sqrt(cache[j]) + eps)
            j += 1
        add_cost(net, costs, valid_costs)
        i += 1
    stop = timeit.default_timer()
    print('Time Gradient: ', stop - start)
    display_results(costs, valid_costs, epochs)
    return 1


def gradient_descent_nes(net, loss='cross_entropy', learning_rate=1.0, batch_size=0, epochs=80):
    costs = []
    valid_costs = []
    i = 0
    start = timeit.default_timer()
    while i < epochs:
        #derivate = backward_pro(net)
        derivate = backward_pro_nes(net)
        j = 0
        #test = copy.deepcopy(net.velocity) # a enlever
        while j < net.size - 1:
            net.velocity[j] = net.momentum * net.velocity[j] - learning_rate * derivate[j]
            #net.thetas[j] = net.thetas[j] - net.momentum * test[j] + ((1 + net.momentum) * net.velocity[j])
            net.thetas[j] = net.thetas[j] + net.velocity[j]
            j += 1
        add_cost(net, costs, valid_costs)
        i += 1
    stop = timeit.default_timer()
    print('Time to perform Nesterov Accelerated Momentum Gradient: ', stop - start)
    display_results(costs, valid_costs, epochs)
    return 1


def stochastic_gradient_descent(net, loss='cross_entropy', learning_rate=1.0, batch_size=0, epochs=80):
    costs = []
    valid_costs = []
    n_batch = len(net.batched_x)
    e = 0
    start = timeit.default_timer()
    while e < epochs:
        b = 0
        while b < n_batch:
            derivate = backward_pro_sto(net, net.batched_x[b], net.batched_vec_y[b])
            j = 0
            while j < net.size - 1:
                net.thetas[j] = net.thetas[j] - learning_rate * derivate[j]
                j += 1
            b += 1
        c = 0
        while c < net.train_size:
            if c < net.valid_size:
                forward_pro(net, net.valid_x[c], train=False)
            forward_pro(net, net.x[c])
            c += 1
        add_cost(net, costs, valid_costs)
        e += 1
    stop = timeit.default_timer()
    print('Time Stochastic Gradient: ', stop - start)
    display_results(costs, valid_costs, epochs)
    return 1


def theta_init(layer_1, layer_2, seed=0, eps=0.5):
    np.random.seed(seed)
    return np.random.rand(layer_2, layer_1 + 1) * 2 * eps - eps


def forward_pro(net, row, train=True):
    activ_dict = {
            'sigmoid': sigmoid,
    }
    i = 0
    a = [row.reshape(-1, 1)]
    b = np.array([[1.0]]).reshape(1, 1)
    while i < net.size - 1:
        a[i] = np.concatenate((b, a[i]), axis=0)
        a.append(activ_dict[net.layers[i].activation](
            net.thetas[i].dot(a[i])))
        i += 1
    if train:
        net.predict.append(a[i])
    else:
        net.valid_predict.append(a[i])
    return a


def forward_pro_sto(net, row):
    activ_dict = {
            'sigmoid': sigmoid,
    }
    i = 0
    a = [row.reshape(-1, 1)]
    b = np.array([[1.0]]).reshape(1, 1)
    while i < net.size - 1:
        a[i] = np.concatenate((b, a[i]), axis=0)
        a.append(activ_dict[net.layers[i].activation](
            net.thetas[i].dot(a[i])))
        i += 1
    return a


def forward_pro_nes(net, row, train=True):
    activ_dict = {
            'sigmoid': sigmoid,
    }
    i = 0
    a = [row.reshape(-1, 1)]
    a_nes = [row.reshape(-1, 1)]
    b = np.array([[1.0]]).reshape(1, 1)
    while i < net.size - 1:
        a[i] = np.concatenate((b, a[i]), axis=0)
        a.append(activ_dict[net.layers[i].activation](
            net.thetas[i].dot(a[i])))
        if train:
            a_nes[i] = np.concatenate((b, a_nes[i]), axis=0)
            a_nes.append(activ_dict[net.layers[i].activation](
                (net.thetas[i] + (net.momentum * net.velocity[i])).dot(a_nes[i])))
        i += 1
    if train:
        net.predict.append(a[i])
    else:
        net.valid_predict.append(a[i])
    return a_nes


def backward_pro(net):
    i = 0
    delta = [0] * (net.size)
    total_delta = copy.deepcopy(net.deltas)
    derivate = [0] * (net.size - 1)
    while i < net.train_size:
        if i < net.valid_size:
            forward_pro(net, net.valid_x[i], train=False)
        a = forward_pro(net, net.x[i])
        j = net.size - 1
        delta[j] = a[j] - net.vec_y[i].reshape(-1, 1)
        j -= 1
        while j > 0:
            delta[j] = net.thetas[j].T.dot(delta[j + 1]) * a[j] * (1 - a[j])
            total_delta[j] += delta[j + 1] * a[j].T
            delta[j] = delta[j][1:, :]
            j -= 1
        total_delta[j] += delta[j + 1] * a[j].T
        i += 1
    i = 0
    while i < net.size - 1:
        if not net.lmbd:
            derivate[i] = total_delta[i] / net.train_size
        else:
            derivate[i] = (total_delta[i] + net.lmbd * net.thetas[i]) # can add to the lasts column direct ? init derivate as np array ?
            derivate[i][:, 0] -= (total_delta[i][:, 0] + net.lmbd * net.thetas[i][:, 0])
            derivate[i] /= net.train_size
        i += 1
    return derivate


def backward_pro_nes(net):
    i = 0
    delta = [0] * (net.size)
    total_delta = copy.deepcopy(net.deltas)
    derivate = [0] * (net.size - 1)
    while i < net.train_size:
        if i < net.valid_size:
            forward_pro_nes(net, net.valid_x[i], train=False)
        a = forward_pro_nes(net, net.x[i])
        j = net.size - 1
        delta[j] = a[j] - net.vec_y[i].reshape(-1, 1)
        j -= 1
        while j > 0:
            delta[j] = (net.thetas[j] + (net.momentum * net.velocity[j])).T.dot(delta[j + 1]) * a[j] * (1 - a[j])
            total_delta[j] += delta[j + 1] * a[j].T
            delta[j] = delta[j][1:, :]
            j -= 1
        total_delta[j] += delta[j + 1] * a[j].T
        i += 1
    i = 0
    while i < net.size - 1:
        if not net.lmbd:
            derivate[i] = total_delta[i] / net.train_size
        else:
            derivate[i] = total_delta[i] + net.lmbd * (net.thetas[i] + (net.momentum * net.velocity[i])) # can add to the lasts column direct ? init derivate as np array ?
            derivate[i][:, 0] -= total_delta[i][:, 0] + (net.lmbd * (net.thetas[i][:, 0] + (net.momentum * net.velocity[i][:, 0])))
            derivate[i] /= net.train_size
        i += 1
    return derivate


def backward_pro_sto(net, x, vec_y):
    i = 0
    delta = [0] * (net.size)
    total_delta = copy.deepcopy(net.deltas)
    derivate = [0] * (net.size - 1)
    batch_size = len(x)
    while i < batch_size:
        a = forward_pro_sto(net, x[i])
        j = net.size - 1
        delta[j] = a[j] - vec_y[i].reshape(-1, 1)
        j -= 1
        while j > 0:
            delta[j] = net.thetas[j].T.dot(delta[j + 1]) * a[j] * (1 - a[j])
            total_delta[j] += delta[j + 1] * a[j].T
            delta[j] = delta[j][1:, :]
            j -= 1
        total_delta[j] += delta[j + 1] * a[j].T
        i += 1
    i = 0
    while i < net.size - 1:
        if not net.lmbd:
            derivate[i] = total_delta[i] / batch_size
        else:
            derivate[i] = (total_delta[i] + net.lmbd * net.thetas[i]) # can add to the lasts column direct ? init derivate as np array ?
            derivate[i][:, 0] -= (total_delta[i][:, 0] + net.lmbd * net.thetas[i][:, 0])
            derivate[i] /= batch_size
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


def cross_entropy(predict, y_class, lmbd, net):
    size = np.size(predict, 0)
    predict = predict.reshape(-1, 2)
    # to do : add counter of class for modularity
    regularization = 0
    if lmbd:
        i = 0
        thetas_sum = 0
        while i < net.size - 1:
            thetas_sum += np.sum(net.thetas[i] ** 2)
            i += 1
        regularization = lmbd / (2 * size) * thetas_sum # can be calc once with attribute in net (else : train + valid)
    # y_0 = (-1 * y_class[:, 0].T.dot((np.log(predict[:, 0]))) - (1 - y_class[:, 0]).T.dot((np.log(1 - predict[:, 0]))))
    # y_1 = (-1 * y_class[:, 1].T.dot((np.log(predict[:, 1]))) - (1 - y_class[:, 1]).T.dot((np.log(1 - predict[:, 1]))))
    # return (1 / size) * (y_0 + y_1) + regularization
    return ((1 / size)
            * (-1 * y_class[:, 0].dot((np.log(predict[:, 0])))
                - (1 - y_class[:, 0]).dot((np.log(1 - predict[:, 0]))))) + regularization


def get_stats(df):
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


def layers_init(hidden_layers, units, n_features, n_class):
    i = 0
    layers = [layer(n_features)]
    while i < hidden_layers:
        layers.append(layer(units))
        i += 1
    layers.append(layer(n_class))
    return layers


def main():
    start = timeit.default_timer()
    df, args = get_data()
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_rows', len(df))
    df = df.rename(columns={0: "id", 1: "class"})
    df = df.drop(columns=['id'])
    # df = df.drop(columns=['id', 20, 13, 11, 16])
    # df = df.drop(columns=['id', 4, 5, 24, 25, 20, 13, 11, 16, 14, 15, 29, 9, 22])
    #df = df.drop(columns=['id', 4, 5, 24, 25, 20, 13, 11, 16, 14, 15, 29, 9, 22, 26, 21, 19, 15, 10, 6])
    stats = get_stats(df)
    df = feature_scaling(df, stats)
    df['class'] = df['class'].map({'M': 1, 'B': 0})
    df['vec_class'] = df['class'].map({1: [1, 0], 0: [0, 1]})
    if args.shuffle:
        df = df.sample(frac=1)
    dfs = np.split(df, [int((len(df) * 0.80))], axis=0)
    if args.outliers:
        df_tmp = dfs[0].copy()
        dfs[0] = dfs[0][(np.abs((df_tmp.select_dtypes(include='number'))) < args.outliers).all(axis=1)]
    layers = layers_init(args.layers, args.units, len(df.columns) - 2, 2)
    net = network(layers, dfs[0], dfs[1], args)
    if args.nesterov:
        gradient_descent_nes(net, learning_rate=args.learning_rate, epochs=args.epochs)
    elif args.rmsprop:
        gradient_descent_rms(net, learning_rate=args.learning_rate, epochs=args.epochs)
    elif args.adagrad:
        gradient_descent_adg(net, learning_rate=args.learning_rate, epochs=args.epochs)
    elif not args.batch_size:
        gradient_descent(net, learning_rate=args.learning_rate, epochs=args.epochs)
    else:
        net.split(args.batch_size)
        stochastic_gradient_descent(net, learning_rate=args.learning_rate, batch_size=args.batch_size, epochs=args.epochs)
    stop = timeit.default_timer()
    print('Time Global: ', stop - start)


if __name__ == '__main__':
    main()
