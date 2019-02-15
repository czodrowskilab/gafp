import os,sys,yaml
import numpy as np
import pandas
from sklearn import metrics
import pathlib


def getKappa(csv_file,y_true,y_preds):
    df = pandas.read_csv(str(csv_file),sep=";", header=0)
    kappas = []
    for y_pred in y_preds:
        kappa = metrics.cohen_kappa_score(df[c1],df[y_pred])
        kappas.append(kappa)
    return kappas



if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Syntax: python {} <predictions.csv> <column_name_true> <column_name_prediction_0> [<column_name_prediction_1> ..]".format(sys.argv[0]),file=sys.stderr)
    csv_file = pathlib.Path(sys.argv[1])
    c1 = sys.argv[2]
    c2 = sys.argv[3:]
    kappas = getKappa(csv_file, c1, c2)
    print("Kappas:")
    for k in kappas:
        print(k)
    print()

    print("Min: {}".format(np.min(kappas)))
    print("Max: {}".format(np.max(kappas)))
    print("Mean: {}".format(np.mean(kappas)))
    print("Median: {}".format(np.median(kappas)))
    print("StdDev: {}".format(np.std(kappas)))