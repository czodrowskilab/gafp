import pickle
from sys import argv

import matplotlib.pyplot as plt
import numpy as np

with open(argv[1], 'rb') as res_file:
    results_df = pickle.load(res_file)

name = argv[2]

max_ix = results_df.kappa_mean.values.argmax()
max_info = results_df.ix[max_ix]

mins = []
maxs = []
for val in results_df.kappa_values:
    mins.append(np.min(val))
    maxs.append(np.max(val))
mins = np.array(mins)
maxs = np.array(maxs)

# Plot with error bars
temp_err = np.row_stack((results_df.kappa_mean - mins, maxs - results_df.kappa_mean))
plt.errorbar(results_df.threshold, results_df.kappa_mean, fmt='none', yerr=temp_err, elinewidth=0.5, capsize=1.5,
             antialiased=True)
plt.plot(results_df.threshold, results_df.kappa_mean, color='C0', marker=None, markersize=4, linewidth=1,
         antialiased=True)
plt.ylabel('Mean Kappa')
plt.xlabel('Variance Threshold')
x_lim = (np.min(results_df.threshold), np.max(results_df.threshold))
y_lim = (plt.ylim()[0], plt.ylim()[1] + 0.025)
plt.xlim(x_lim)
plt.ylim(y_lim)
xytext = (max_info.threshold + 0.005, max_info.kappa_mean + 0.025)
if name == 'solubility':
    xytext = (max_info.threshold + 0.005, max_info.kappa_mean + 0.015)
plt.annotate(' thresh: %.3f \n'
             ' kappa_mean: %.6f \n'
             ' kappa_std: %.6f \n'
             ' n_bits: %i' % (max_info.threshold,
                              max_info.kappa_mean,
                              max_info.kappa_std,
                              max_info.n_bits),
             xy=(max_info.threshold, max_info.kappa_mean),
             xytext=xytext,
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.savefig(f'{name}_errorbar.png', dpi=400)
plt.clf()

# Plot like in Chi's master thesis
plt.plot(results_df.threshold, results_df.kappa_mean, antialiased=True)
plt.ylabel('Mean Kappa')
plt.xlabel('Variance Threshold')
plt.xlim(x_lim)
plt.ylim(y_lim)
xytext = (max_info.threshold + 0.005, max_info.kappa_mean + 0.005)
plt.annotate(' thresh: %.3f \n'
             ' kappa_mean: %.6f \n'
             ' kappa_std: %.6f \n'
             ' n_bits: %i' % (max_info.threshold,
                              max_info.kappa_mean,
                              max_info.kappa_std,
                              max_info.n_bits),
             xy=(max_info.threshold, max_info.kappa_mean),
             xytext=xytext,
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.savefig(f'{name}_default.png', dpi=400)
plt.clf()

# Plot with nbits on y axis
plt.plot(results_df.threshold, results_df.n_bits, antialiased=True)
plt.ylabel('n Bits')
plt.xlabel('Variance Threshold')
plt.xlim(x_lim)
plt.ylim(plt.ylim()[0], plt.ylim()[1] + 0.025)
xytext = (max_info.threshold + 0.005, max_info.n_bits + 300)
if name == 'clint_rat':
    xytext = (max_info.threshold + 0.01, max_info.n_bits - 600)
plt.annotate(' thresh: %.3f \n'
             ' kappa_mean: %.6f \n'
             ' kappa_std: %.6f \n'
             ' n_bits: %i' % (max_info.threshold,
                              max_info.kappa_mean,
                              max_info.kappa_std,
                              max_info.n_bits),
             xy=(max_info.threshold, max_info.n_bits),
             xytext=xytext,
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.savefig(f'{name}_nbits.png', dpi=400)
plt.clf()
