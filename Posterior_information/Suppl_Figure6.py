
#### import libraries

import numpy as np
import pylab as pl
import pandas as pd
import random

##################################################################################
### uploading information from Monte Carlo inference

df = pd.read_csv('chain_Friedman.csv') 
chain_new = np.array(df)

sample_chain_new = random.sample(list(chain_new), 5000)
sample_chain_new = np.array(sample_chain_new)

##############################################################
### plotting marginal distribution for growth rates

fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = pl.subplots(2, 4,figsize= (12,6))
ax1.hist(sample_chain_new[:,0],bins=15,density =True)
ax1.set_title(r'$\theta_1$')
ax1.set_xlim([0.02, 0.2])
ax1.set_ylim([0, 550])
ax1.label_outer()
ax2.hist(sample_chain_new[:,9],bins=15,density =True)
ax2.set_title(r'$\theta_2$')
ax2.set_xlim([0.02, 0.2])
ax2.set_ylim([0, 550])
ax2.label_outer()
ax3.hist(sample_chain_new[:,18],bins=15,density =True)
ax3.set_title(r'$\theta_3$')
ax3.set_xlim([0.02, 0.2])
ax3.set_ylim([0, 550])
ax3.label_outer()
ax4.hist(sample_chain_new[:,27],bins=15,density =True)
ax4.set_title(r'$\theta_4$')
ax4.set_xlim([0.02, 0.2])
ax4.set_ylim([0, 550])
ax4.label_outer()
ax5.hist(sample_chain_new[:,36],bins=15,density =True)
ax5.set_title(r'$\theta_5$')
ax5.set_xlim([0.02, 0.2])
ax5.set_ylim([0, 550])
ax6.hist(sample_chain_new[:,45],bins=15,density =True)
ax6.set_title(r'$\theta_6$')
ax6.set_xlim([0.02, 0.2])
ax6.set_ylim([0, 550])
ax6.label_outer()
ax7.hist(sample_chain_new[:,54],bins=15,density =True)
ax7.set_title(r'$\theta_7$')
ax7.set_xlim([0.02, 0.2])
ax7.set_ylim([0, 550])
ax7.label_outer()
ax8.hist(sample_chain_new[:,63],bins=15,density =True)
ax8.set_title(r'$\theta_8$')
ax8.set_xlim([0.02, 0.2])
ax8.set_ylim([0, 550])
ax8.label_outer()
fig.tight_layout()
pl.savefig('growth_Friedman.pdf')
pl.show()

##############################################################
