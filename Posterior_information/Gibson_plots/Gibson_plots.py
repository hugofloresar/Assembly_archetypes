import scipy.stats as ss
import numpy as np
import pandas as pd
import pylab as pl


healthy = 'h'
ulcerative_col = 'uc'


# type_pat = healthy
type_pat = ulcerative_col

samplesr = pd.read_csv('r'+type_pat+'.csv', header = None)
rs = np.array(samplesr)

A = pd.read_csv('A'+type_pat+'.csv', header = None)
A = np.array(A)

B = np.empty((16,16))
for i in range(16):
    B[:,i] = A[:,(15-i)]


import matplotlib.pyplot as plt
import matplotlib.colors as colors


# plt.figure(figsize = (20,8))
# bounds = np.array([np.min(A),-0.1, 0, 0.1, -np.min(A)])
# norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
# plt.pcolor(A, norm=norm, cmap='RdBu_r')
# for j in range(16):
#     for i in range(16):
#         if A[j][i]<-0.1:
#             # pl.text(i + 0.5, j + 0.5, '%.2f' % A[j][i], color='white',weight='bold', ha='center', va='center')
#             pl.text(i + 0.5, j + 0.5, f'{A[j][i]:.1e}', color='white',weight='bold', ha='center', va='center')
#         elif A[j][i]>0.1:
#             # pl.text(i + 0.5, j + 0.5, '%.2f' % A[j][i], color='white',weight='bold', ha='center', va='center')
#             pl.text(i + 0.5, j + 0.5, f'{A[j][i]:.1e}', color='white',weight='bold', ha='center', va='center')
#         else:
#             # pl.text(i + 0.5, j + 0.5, '%.2f' % A[j][i], ha='center', va='center')
#             pl.text(i + 0.5, j + 0.5, f'{A[j][i]:.1e}', color='white',weight='bold', ha='center', va='center')
# plt.colorbar(extend='both', orientation='vertical')
# plt.title('Interactions case '+type_pat)
# pl.xticks(np.array([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,
#                     11.5,12.5,13.5,14.5,15.5])
#           ,['OTU_1', 'OTU_2', 'OTU_3', 'OTU_4', 'OTU_5', 'OTU_6' ,
#             'OTU_7', 'OTU_8','OTU_9','OTU_10','OTU_12','OTU_13','OTU_16',
#             'OTU_17','OTU_18','OTU_21'])
# pl.yticks(np.array([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,
#                     11.5,12.5,13.5,14.5,15.5])
#           ,['OTU_1', 'OTU_2', 'OTU_3', 'OTU_4', 'OTU_5', 'OTU_6' ,
#             'OTU_7', 'OTU_8','OTU_9','OTU_10','OTU_12','OTU_13','OTU_16',
#             'OTU_17','OTU_18','OTU_21'])
# pl.savefig('A'+type_pat+'.pdf')
# pl.show()

plt.figure(figsize = (20,8))
bounds = np.array([np.min(A),-0.1, 0, 0.1, -np.min(A)])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
plt.pcolor(B, norm=norm, cmap='RdBu_r')
for j in range(16):
    for i in range(16):
        if B[j][i]<-0.1:
            pl.text(i + 0.5, j + 0.5, f'{B[j][i]:.1e}', color='white',weight='bold', ha='center', va='center')
        elif B[j][i]>0.1:
            pl.text(i + 0.5, j + 0.5, f'{B[j][i]:.1e}', color='white',weight='bold', ha='center', va='center')
        else:
            pl.text(i + 0.5, j + 0.5, f'{B[j][i]:.1e}', ha='center', va='center')
plt.colorbar(extend='both', orientation='vertical')
plt.title('Interactions case '+type_pat, size = 18)
pl.xticks(np.array([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,
                    11.5,12.5,13.5,14.5,15.5])
          ,['OTU_21', 'OTU_18', 'OTU_17', 'OTU_16', 'OTU_13', 'OTU_12' ,
            'OTU_10', 'OTU_9','OTU_8','OTU_7','OTU_6','OTU_5','OTU_4',
            'OTU_3','OTU_2','OTU_1'])
pl.yticks(np.array([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,
                    11.5,12.5,13.5,14.5,15.5])
          ,['OTU_1', 'OTU_2', 'OTU_3', 'OTU_4', 'OTU_5', 'OTU_6' ,
            'OTU_7', 'OTU_8','OTU_9','OTU_10','OTU_12','OTU_13','OTU_16',
            'OTU_17','OTU_18','OTU_21'])
pl.savefig('B'+type_pat+'.pdf')
pl.show()


########################################################
#### Both posterior simultaneously

samplesh = pd.read_csv('rh.csv', header = None)
rsh = np.array(samplesh)

samplesuc = pd.read_csv('ruc.csv', header = None)
rsuc = np.array(samplesuc)


fig, ((ax1, ax2, ax3, ax4),(ax5, ax6, ax7, ax8),(ax9, ax10, ax11, ax12),(ax13, ax14, ax15, ax16)) = pl.subplots(4, 4,figsize= (12,6))
ax1.hist(rsh[:,0],bins=15,alpha = 0.5,density =True)
ax1.hist(rsuc[:,0],bins=15,alpha = 0.5,density =True)
ax1.set_xlim([0, 1.5])
ax1.set_ylim([0, 4])
ax1.set_title(r'$\theta_1$')
ax2.hist(rsh[:,1],bins=15,alpha = 0.5,density =True)
ax2.hist(rsuc[:,1],bins=15,alpha = 0.5,density =True)
ax2.set_xlim([0, 1.5])
ax2.set_ylim([0, 4])
ax2.set_title(r'$\theta_2$')
ax3.hist(rsh[:,2],bins=15,alpha = 0.5,density =True)
ax3.hist(rsuc[:,2],bins=15,alpha = 0.5,density =True)
ax3.set_xlim([0, 1.5])
ax3.set_ylim([0, 4])
ax3.set_title(r'$\theta_3$')
ax4.hist(rsh[:,3],bins=15,alpha = 0.5,density =True)
ax4.hist(rsuc[:,3],bins=15,alpha = 0.5,density =True)
ax4.set_xlim([0, 1.5])
ax4.set_ylim([0, 4])
ax4.set_title(r'$\theta_4$')
ax5.hist(rsh[:,4],bins=15,alpha = 0.5,density =True)
ax5.hist(rsuc[:,4],bins=15,alpha = 0.5,density =True)
ax5.set_xlim([0, 1.5])
ax5.set_ylim([0, 4])
ax5.set_title(r'$\theta_5$')
ax6.hist(rsh[:,5],bins=15,alpha = 0.5,density =True)
ax6.hist(rsuc[:,5],bins=15,alpha = 0.5,density =True)
ax6.set_xlim([0, 1.5])
ax6.set_ylim([0, 4])
ax6.set_title(r'$\theta_6$')
ax7.hist(rsh[:,6],bins=15,alpha = 0.5,density =True)
ax7.hist(rsuc[:,6],bins=15,alpha = 0.5,density =True)
ax7.set_xlim([0, 1.5])
ax7.set_ylim([0, 4])
ax7.set_title(r'$\theta_7$')
ax8.hist(rsh[:,7],bins=15,alpha = 0.5,density =True)
ax8.hist(rsuc[:,7],bins=15,alpha = 0.5,density =True)
ax8.set_xlim([0, 1.5])
ax8.set_ylim([0, 4])
ax8.set_title(r'$\theta_8$')
ax9.hist(rsh[:,8],bins=15,alpha = 0.5,density =True)
ax9.hist(rsuc[:,8],bins=15,alpha = 0.5,density =True)
ax9.set_xlim([0, 1.5])
ax9.set_ylim([0, 4])
ax9.set_title(r'$\theta_9$')
ax10.hist(rsh[:,9],bins=15,alpha = 0.5,density =True)
ax10.hist(rsuc[:,9],bins=15,alpha = 0.5,density =True)
ax10.set_xlim([0, 1.5])
ax10.set_ylim([0, 4])
ax10.set_title(r'$\theta_{10}$')
ax11.hist(rsh[:,10],bins=15,alpha = 0.5,density =True)
ax11.hist(rsuc[:,10],bins=15,alpha = 0.5,density =True)
ax11.set_xlim([0, 1.5])
ax11.set_ylim([0, 4])
ax11.set_title(r'$\theta_{11}$')
ax12.hist(rsh[:,11],bins=15,alpha = 0.5,density =True)
ax12.hist(rsuc[:,11],bins=15,alpha = 0.5,density =True)
ax12.set_xlim([0, 1.5])
ax12.set_ylim([0, 4])
ax12.set_title(r'$\theta_{12}$')
ax13.hist(rsh[:,12],bins=15,alpha = 0.5,density =True)
ax13.hist(rsuc[:,12],bins=15,alpha = 0.5,density =True)
ax13.set_xlim([0, 1.5])
ax13.set_ylim([0, 4])
ax13.set_title(r'$\theta_{13}$')
ax14.hist(rsh[:,13],bins=15,alpha = 0.5,density =True)
ax14.hist(rsuc[:,13],bins=15,alpha = 0.5,density =True)
ax14.set_xlim([0, 1.5])
ax14.set_ylim([0, 4])
ax14.set_title(r'$\theta_{14}$')
ax15.hist(rsh[:,14],bins=15,alpha = 0.5,density =True)
ax15.hist(rsuc[:,14],bins=15,alpha = 0.5,density =True)
ax15.set_xlim([0, 1.5])
ax15.set_ylim([0, 4])
ax15.set_title(r'$\theta_{15}$')
ax16.hist(rsh[:,15],bins=15,alpha = 0.5,density =True)
ax16.hist(rsuc[:,15],bins=15,alpha = 0.5,density =True)
ax16.set_xlim([0, 1.5])
ax16.set_ylim([0, 4])
ax16.set_title(r'$\theta_{16}$')
fig.tight_layout()
pl.legend(('Healthy', 'UC'))
pl.savefig('rs_both.pdf')
pl.show()


######################################################
