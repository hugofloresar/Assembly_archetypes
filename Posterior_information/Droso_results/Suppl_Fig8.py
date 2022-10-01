import numpy as np
import scipy as sp
import pandas as pd
import sys
import csv
import os
import pylab as pl



samples = pd.read_csv('samples_GLV_Droso.csv',header = None)
samples = np.array(samples)


##### Marginal Posterior for intrinsic growth

fig, (ax1, ax2, ax3,ax4,ax5) = pl.subplots(1, 5,figsize= (12,3))
ax1.hist(samples[:,0],bins=15,density =True)
ax1.set_title(r'$\theta_1$')
ax1.set_xlim([0, 0.25])
ax1.set_ylim([0, 30])
ax2.hist(samples[:,1],bins=15,density =True)
ax2.set_title(r'$\theta_2$')
ax2.set_xlim([0, 0.25])
ax2.set_ylim([0, 30])
ax2.label_outer()
ax3.hist(samples[:,2],bins=15,density =True)
ax3.set_title(r'$\theta_3$')
ax3.set_xlim([0, 0.25])
ax3.set_ylim([0, 30])
ax3.label_outer()
ax4.hist(samples[:,3],bins=15,density =True)
ax4.set_title(r'$\theta_4$')
ax4.set_xlim([0, 0.25])
ax4.set_ylim([0, 30])
ax4.label_outer()
ax5.hist(samples[:,4],bins=15,density =True)
ax5.set_title(r'$\theta_5$')
ax5.set_xlim([0, 0.25])
ax5.set_ylim([0, 30])
ax5.label_outer()
fig.tight_layout()
pl.show()


###########################################################
MAPall = pd.read_csv('Amap.csv',header = None)
A = np.array(MAPall)

B = np.empty((5,5))
for i in range(5):
    B[:,i] = A[:,(4-i)]

import matplotlib.pyplot as plt
import matplotlib.colors as colors


plt.figure(figsize = (8,3))
bounds = np.array([np.min(A),-1.0e-07,0.0,1.0e-07, -np.min(A)])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
plt.pcolor(B, norm=norm, cmap='RdBu_r')
for j in range(5):
    for i in range(5):
        if B[j][i]<-1.0e-07:
            pl.text(i + 0.5, j + 0.5, f'{B[j][i]:.1e}', color='white',weight='bold', ha='center', va='center')
        elif B[j][i]>1.0e-07:
            pl.text(i + 0.5, j + 0.5, f'{B[j][i]:.1e}', color='white',weight='bold', ha='center', va='center')
        else:
            pl.text(i + 0.5, j + 0.5, f'{B[j][i]:.1e}', ha='center', va='center')
plt.colorbar(extend='both', orientation='vertical')
pl.xticks(np.array([0.5,1.5,2.5,3.5,4.5])
          ,["AO","AT","AP","LB","LP"])
pl.yticks(np.array([0.5,1.5,2.5,3.5,4.5])
          ,["LP","LB","AP","AT","AO"])
pl.show()



