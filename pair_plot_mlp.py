import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def main():
    df = get_data(sys.argv)
    sns.set(font_scale=0.5)
    sns.pairplot(
                    df.drop(columns=[0, 4, 5, 24, 25, 20, 13, 11, 16, 14, 15, 29, 9, 22]).dropna(),
                    hue=1,
                    height=2,
                    aspect=1)
    plt.subplots_adjust(left=0.04, bottom=0.04)
    plt.show()


if __name__ == '__main__':
    main()
