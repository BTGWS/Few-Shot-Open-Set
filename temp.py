import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('bmh')
matplotlib.rcParams.update({'errorbar.capsize': 2})
# plt.rcParams.update({'font.size': 14})

lmbda1 = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
## gtsrb 5_way
auroc_11 = np.array([72.00,79.78, 85.25, 82.02, 80.25])
err_11 = np.array([0.68, 0.61, 0.56, 0.55, 0.72])
auroc_11_ = np.array([64.17, 75.34, 81.98, 78.96, 79.01])
err_11_ = np.array([0.90, 0.67, 0.68, 0.63, 0.58])
## belga
auroc_12 = np.array([54.74, 70.73, 74.61, 73.30, 76.12])
err_12 = np.array([0.92, 0.84, 0.82, 0.81, 0.79])


fig = plt.figure()
ax = fig.add_subplot(111)

ax.errorbar(lmbda1,auroc_11,yerr=err_11,fmt='-o')
ax.errorbar(lmbda1,auroc_11_,yerr=err_11_,fmt='-o')
ax.set_yticks([60,65,70,75,80,85])
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.set_xscale('log')
ax.set_ylim([60,90])
ax.set_xlabel(r'$\lambda_1$')
ax.set_ylabel('AUROC')
ax.legend(['5-way 5-shot', '5-way 1-shot'])
plt.savefig('lmbda_1.pdf', format='pdf', dpi=1200, bbox_inches = 'tight', pad_inches = 0)
plt.show()


lmbda3 = np.array([1,5,10,100])
fig = plt.figure()
ax = fig.add_subplot(111)

## gtsrb 5shot
acc_31 = np.array([83.05, 82.99, 81.66, 80.29])
auroc_31 = np.array([83.95, 85.04, 85.25, 76.45])
err_acc_31 = np.array([0.73, 0.73, 0.77, 0.83])
err_auroc_31 = np.array([0.53, 0.51, 0.56, 0.62])
## gtsrb 1shot
acc_31_ = np.array([72.74, 70.08, 72.89, 68.23])
auroc_31_ = np.array([76.34, 81.58, 81.98, 76.45])
err_acc_31_ = np.array([0.89, 0.90, 0.91, 0.93])
err_auroc_31_ = np.array([0.64, 0.54, 0.59, 0.62])
ax.errorbar(lmbda3,auroc_31,yerr=err_auroc_31,fmt='-o')
ax.errorbar(lmbda3,auroc_31_,yerr=err_auroc_31_,fmt='-o')
ax.set_yticks([60,65,70,75,80,85])
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.set_xscale('log')
ax.set_ylim([60,90])
ax.set_xlabel(r'$\lambda_3$')
ax.set_ylabel('AUROC')
ax.legend(['5-way 5-shot', '5-way 1-shot'])
plt.savefig('lmbda_3.pdf', format='pdf', dpi=1200, bbox_inches = 'tight', pad_inches = 0)
plt.show()