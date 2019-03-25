import sys
import inflect
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
    return df.loc[df['Hogwarts House'] == house, subject].dropna()


def f_test(huf, gry, rav, sly):
    huf_size = huf.size
    gry_size = gry.size
    rav_size = rav.size
    sly_size = sly.size
    total_size = huf_size + gry_size + rav_size + sly_size

    huf_mean = np.sum(huf) / huf_size
    gry_mean = np.sum(gry) / gry_size
    rav_mean = np.sum(rav) / rav_size
    sly_mean = np.sum(sly) / sly_size
    total_mean = (huf_mean + gry_mean + rav_mean + sly_mean) / 4
    SSWG = (
            np.sum(np.power(huf - huf_mean, 2))
            + np.sum(np.power(gry - gry_mean, 2))
            + np.sum(np.power(rav - rav_mean, 2))
            + np.sum(np.power(sly - sly_mean, 2)))
    SSBG = (
            np.power(huf_mean - total_mean, 2) * huf_size
            + np.power(gry_mean - total_mean, 2) * gry_size
            + np.power(rav_mean - total_mean, 2) * rav_size
            + np.power(sly_mean - total_mean, 2) * sly_size)
    return (SSBG / 3) / (SSWG / (total_size - 4))


def auto_selection(df, houses, subjects):
    F = {}
    for subject in subjects:
        F[subject] = f_test(
                get_subject_grade_house(df, subject, 'Hufflepuff').values,
                get_subject_grade_house(df, subject, 'Gryffindor').values,
                get_subject_grade_house(df, subject, 'Ravenclaw').values,
                get_subject_grade_house(df, subject, 'Slytherin').values)
    # https://web.ma.utexas.edu/users/davis/375/popecol/tables/f005.html
    critical_value = 2.61
    F = {k: v for k, v in sorted(
        F.items(), key=lambda x: x[1], reverse=False)}
    F = {k: v for k, v in F.items() if v <= critical_value}
    homogens_subjects = list(F.keys())
    p = inflect.engine()
    i = 1
    ifig = 3
    for homogen_subject in homogens_subjects:
        plt.figure(ifig)
        for house in houses:
            grade = df.loc[
                    df['Hogwarts House'] == house, homogen_subject].dropna()
            plt.hist(grade, alpha=0.5, label=house)
        plt.legend(loc='upper right')
        plt.xlabel(homogen_subject + ' Grades')
        plt.ylabel('Frequency')
        plt.title(
                    p.ordinal(i)
                    + " homogeneous distribution: "
                    + homogen_subject
                    + " (Auto Selection)")
        df.boxplot(homogen_subject, by='Hogwarts House', figsize=(12, 8))
        plt.ylabel(homogen_subject + ' Grades')
        plt.show()
        ifig += 2
        i += 1


def manual_selection(df, houses):
    plt.figure(1)
    for house in houses:
        grade = df.loc[
                df['Hogwarts House'] == house, 'Care of Magical Creatures'
                ].dropna()
        plt.hist(grade, alpha=0.5, label=house)
    plt.legend(loc='upper right')
    plt.xlabel('Care of Magical Creatures Grades')
    plt.ylabel('Frequency')
    plt.title(
                "Most homogeneous distribution: "
                + "Care of Magical Creatures (Manual Selection)")
    df.boxplot(
                'Care of Magical Creatures',
                by='Hogwarts House',
                figsize=(12, 8))
    plt.ylabel('Care of Magical Creatures Grades')
    plt.show()


def select_features(df):
    houses = df['Hogwarts House'].unique().tolist()
    subjects = list(df.select_dtypes('number').to_dict().keys())
    subjects.remove('Index')
    return houses, subjects


def main():
    df = get_data(sys.argv)
    houses, subjects = select_features(df)
    manual_selection(df, houses)
    auto_selection(df, houses, subjects)


if __name__ == '__main__':
    main()
