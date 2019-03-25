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


def theta_init(layer_1, layer_2, e=0.5):
    return np.random.rand(layer_2, layer_1 + 1) * 2 * e - e


def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))


def forward_pro(theta, a, activation='sigmoid'):
    activ_dict = {
            'sigmoid': sigmoid,
    }
    return activ_dict[activation](theta * a)


def softmax(h):
    return np.exp(-1 * h) / (np.sum(np.exp(-1 * h)))


def main():
    pd.set_option('display.expand_frame_repr', False)
    df = get_data(sys.argv)
    pd.set_option('display.max_rows', len(df))
    df = df.rename(columns={0: "id", 1: "Class"})
    describe(forward_pro(2, 3))
    # print(df.describe())
    # print(df)


if __name__ == '__main__':
    main()
