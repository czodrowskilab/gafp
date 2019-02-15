from sys import argv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

val_file = argv[1]
err_min_file = argv[2]
err_max_file = argv[3]

bar_width = 0.15
gap = 0.05

vals = pd.read_csv(val_file, sep=';', index_col=False)
err_min = pd.read_csv(err_min_file, sep=';', index_col=False)
err_max = pd.read_csv(err_max_file, sep=';', index_col=False)

index = np.arange(len(vals.Dataset))
cols = ['Standard RF',
        'Standard DNN',
        'Standard XGB',
        'GA DNN']

error_config = dict(elinewidth=0.5, capsize=1.5)
colors = ['gold',
          'limegreen',
          'firebrick',
          'darkgreen']

ix = 0
for col in cols:
    err = []
    for i in index:
        err.append((vals[col][i] - err_min[col][i], err_max[col][i] - vals[col][i]))
    err = np.transpose(np.row_stack(err))
    plt.bar(index + gap * ix + bar_width * ix, vals[col], bar_width,
            label=col.replace('Standard ', ''), antialiased=True, color=colors[ix],
            yerr=err, error_kw=error_config)
    ix += 1

plt.ylabel('Mean Kappa')
plt.xlabel('Endpoint')
plt.xticks(index + bar_width * 3 / 1.5, vals.Dataset)
plt.legend()

plt.savefig('last_plot.png', dpi=400)
