from os.path import basename
from sys import argv

from rdkit.Chem.PandasTools import LoadSDF
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

files = argv[2:]
thres = {'clint_human': (30, 100),
         'clint_mouse': (30, 100),
         'clint_rat': (30, 100),
         'caco': (3, 20),
         'herg': (5,),
         'solu': (-5, -4)}
cats = {'clint_human': ('stable', 'intermediate', 'instable'),
        'clint_mouse': ('stable', 'intermediate', 'instable'),
        'clint_rat': ('stable', 'intermediate', 'instable'),
        'caco': ('impermeable', 'intermediate', 'permeable'),
        'herg': ('inactive', 'active'),
        'solu': ('insoluble', 'intermediate', 'soluble')}
tags = {'clint_human': 'clint',
        'clint_mouse': 'clint',
        'clint_rat': 'clint',
        'caco': 'papp',
        'herg': 'pKi',
        'solu': 'logS'}


def categorize(val, key_val):
    cat = None
    for i in range(len(thres[key_val])):
        if val < thres[key_val][i]:
            cat = cats[key_val][i]
            break
    if not cat:
        cat = cats[key_val][-1]
    return cat


with open(argv[1], 'w') as outfile:
    for file in files:
        key = basename(file).split('.')[0]
        df = LoadSDF(file)
        pred_col = f'{key}_prediction'
        if pred_col not in df.columns:
            pred_col = [col for col in df.columns if '_prediction' in col][0]
        y_pred = df[pred_col]
        y_true_col = tags[key]
        y_true_uncat = df[y_true_col]
        y_true = y_true_uncat.astype(float).apply(categorize, args=(key,))
        kappa = cohen_kappa_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        outfile.write(f'{key}\n')
        outfile.write('=============\n')
        outfile.write(f'Acc: {acc}\n')
        outfile.write(f'Kappa: {kappa}\n')
        outfile.write(f'TP: {cm[1,1]}, FP: {cm[0,1]}, TN: {cm[0,0]}, FN: {cm[1,0]}\n\n')
