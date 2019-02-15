import pickle
from sys import argv

import numpy as np
from sklearn.metrics import cohen_kappa_score

from ctrainlib.fplib import compute_fingerprints
from ctrainlib.rdkit_support import compute_descriptors
from ctrainlib.rdkit_support import load_sdf_as_dataframe

thresholds = {'clint': [30, 100],
              'pKi': [5],
              'papp': [3, 20],
              'logS': [-5, -4]}
value_col = argv[3]


def categorize(val):
    cat = None
    cats = list(range(len(thresholds[value_col]) + 1))
    for i in range(len(thresholds[value_col])):
        if val < thresholds[value_col][i]:
            cat = cats[i]
            break
    if cat is None:
        cat = cats[-1]
    return cat


with open(argv[1], 'rb') as pkl:
    model = pickle.load(pkl)
df = load_sdf_as_dataframe(argv[2], keep_props=True, value_tag=value_col)
df = compute_descriptors(df, model.descriptors)
df = compute_fingerprints(df, model.fingerprints)
df_to_pred = df.drop(columns=['ROMol', value_col])

true_val = list(map(categorize, np.array(df[value_col], float)))

print('##########################')

kappas = []
for m in model.model.models:
    probs = m.predict_proba(np.array(df_to_pred))
    if len(argv) > 4:
        classes = list(range(len(thresholds[value_col]) + 1))
        pred = np.array([classes[x] for x in list(np.argmax(probs, axis=1))])
    else:
        pred = np.array([m.classes_[x] for x in list(np.argmax(probs, axis=1))])
    kappas.append(cohen_kappa_score(true_val, pred))
    print(kappas[-1])
print()
print(kappas)
print(f'Mean: {np.mean(kappas)}')
print(f'Min: {np.min(kappas)}')
print(f'Max: {np.max(kappas)}')

print('##########################')
