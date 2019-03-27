import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def get_subject_grade_house(df, subject, house):
    return df.loc[df[1] == house, subject].dropna()


def f_test(huf, gry):
    huf_size = huf.size
    gry_size = gry.size
    total_size = huf_size + gry_size

    huf_mean = np.sum(huf) / huf_size
    gry_mean = np.sum(gry) / gry_size
    total_mean = (huf_mean + gry_mean) / 2
    SSWG = (
            np.sum(np.power(huf - huf_mean, 2))
            + np.sum(np.power(gry - gry_mean, 2)))
    SSBG = (
            np.power(huf_mean - total_mean, 2) * huf_size
            + np.power(gry_mean - total_mean, 2) * gry_size)
    return (SSBG / 1) / (SSWG / (total_size - 2))


def auto_selection(df, houses, subjects):
    F = {}
    for subject in subjects:
        F[subject] = f_test(
                get_subject_grade_house(df, subject, 'M').values,
                get_subject_grade_house(df, subject, 'B').values,
                )
    # https://web.ma.utexas.edu/users/davis/375/popecol/tables/f005.html
    critical_value = 3.00
    F = {k: v for k, v in sorted(
        F.items(), key=lambda x: x[1], reverse=False)}
    print(F)
    F = {k: v for k, v in F.items() if v <= critical_value}
    homogens_subjects = list(F.keys())
    i = 1
    ifig = 3
    for homogen_subject in homogens_subjects:
        plt.figure(ifig)
        for house in houses:
            grade = df.loc[
                    df[1] == house, homogen_subject].dropna()
            plt.hist(grade, alpha=0.5, label=house)
        plt.legend(loc='upper right')
        plt.xlabel(str(homogen_subject) + ' Grades')
        plt.ylabel('Frequency')
        plt.show()
        ifig += 2
        i += 1




def select_features(df):
    houses = df[1].unique().tolist()
    subjects = list(df.select_dtypes('number').to_dict().keys())
    subjects.remove(0)
    return houses, subjects


def main():
    df = get_data(sys.argv)
    houses, subjects = select_features(df)
    auto_selection(df, houses, subjects)


if __name__ == '__main__':
    main()
