import sys
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_data(args):
    file = "data.csv"
    if len(args) < 2:
        print("No arguments given. Try with \"data.csv\".")
    else:
        file = args[1]
    try:
        data = pd.read_csv(file, header=None)
    except Exception as e:
        print("Can't extract data from {}.".format(file))
        print(e.__doc__)
        sys.exit(0)
    return data


def var_init(df):
    houses = df[1].values
    colors = {
                'M': 'Red',
                'B': 'Green'}
    return houses, colors


def pearson_correlation(df):
    feat1 = df.iloc[:, 0].values
    feat2 = df.iloc[:, 1].values
    size = feat1.size
    sum1 = np.sum(feat1)
    sum2 = np.sum(feat2)
    mean1 = sum1 / size
    mean2 = sum2 / size
    std1 = np.sqrt(np.sum(np.power(feat1 - mean1, 2)) / (size - 1))
    std2 = np.sqrt(np.sum(np.power(feat2 - mean2, 2)) / (size - 1))
    return ((np.sum(feat1 * feat2) - size * mean1 * mean2)
            / ((size - 1) * std1 * std2))


def plot_scatter(df, feat_cor, houses, colors, sign):
    scatter_x = df[feat_cor[0]].values
    scatter_y = df[feat_cor[1]].values
    fig, ax = plt.subplots()
    for house in np.unique(houses):
        ix = np.where(houses == house)
        ax.scatter(
                    scatter_x[ix], scatter_y[ix],
                    c=colors[house], label=house, alpha=0.5)
    plt.xlabel(feat_cor[0])
    plt.ylabel(feat_cor[1])
    plt.title('Strongest ' + sign + ' Correlation')
    ax.legend()


def auto_selection(df, houses, colors):
    subjects = list(df.select_dtypes('number').to_dict().keys())
    subjects.remove(0)
    subjects2 = subjects.copy()
    cor = {}
    for sub in subjects:
        subjects2.pop(0)
        for sub2 in subjects2:
            cor[sub, sub2] = pearson_correlation(df[[sub, sub2]].dropna())
    sorted_cor = sorted(cor.items(), key=operator.itemgetter(1))
    print(sorted_cor)
    feat_cor_max = max(cor.items(), key=operator.itemgetter(1))[0]
    feat_cor_min = min(cor.items(), key=operator.itemgetter(1))[0]
    val_cor_max = max(cor.items(), key=operator.itemgetter(1))[1]
    val_cor_min = min(cor.items(), key=operator.itemgetter(1))[1]
    if val_cor_max > 0:
        plot_scatter(df, feat_cor_max, houses, colors, 'Positive')
    if val_cor_min < 0:
        plot_scatter(df, feat_cor_min, houses, colors, 'Negative')


def main():
    df = get_data(sys.argv)
    houses, colors = var_init(df)
    auto_selection(df, houses, colors)
    plt.show()


if __name__ == '__main__':
    main()
