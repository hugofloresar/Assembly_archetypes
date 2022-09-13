
import numpy as np
import pandas as pd

healthy = 'h'
ulcerative_col = 'uc'



### choose the kind of patient
# type_pat = healthy
type_pat = ulcerative_col



samplesr = pd.read_csv('r_samples_'+type_pat+'.csv', header = None)
rhealthy = np.array(samplesr)

cluster_valueh = pd.read_csv('cluster_value_'+type_pat+'.csv', header = None)
cluster_valueh = np.array(cluster_valueh)

# if type_pat == healthy:
#     cluster_valueh = cluster_valueh[0]
# else:
cluster_valueh = cluster_valueh.T[0]


# cluster_assignh = pd.read_csv('cluster_assignment_'+type_pat+'.csv', header = None)
# cluster_assignh = np.array(cluster_assignh,dtype = int)
# cluster_assignh = cluster_assignh.T[0]

adiag_mean_h = pd.read_csv('adiag_mean_'+type_pat+'.csv', header = None)
adiag_mean_h = np.array(adiag_mean_h)
adiag_mean_h = adiag_mean_h.T[0]

##################################################################################
dim = len(adiag_mean_h)

A = np.zeros((dim, dim))

#### fill the diagonal
for i in range(dim):
    A[i][i] = -adiag_mean_h[i]


#### fill the nondiagonal elements
if type_pat == healthy:
    loc0 = [15,16,19]
    loc1 = [0,1,2,3,4,5,6,7,8,10,11,14,18]
    loc2 = [9,12,13,17]
elif type_pat == ulcerative_col:
    loc0 = [3,5,6,12,13,14,16,17,19]
    loc1 = [0,1,7,8,9,10,11,15,18]
    loc2 = [2,4]
    

#### For healthy case, there is 3 clusters
#### For ulcerative case, there is 1 cluster
if type_pat == healthy:
    for i in loc0:
        for j in loc1:
            A[i][j] = cluster_valueh[0]
    for i in loc0:
        for jj in loc2:
            A[jj][i] = cluster_valueh[1]
    for i in loc1:
        for jj in loc2:
            A[jj][i] = cluster_valueh[2]
elif type_pat == ulcerative_col:
    for i in loc0:
        for j in loc1:
            A[i][j] = cluster_valueh[0]


if type_pat ==healthy:
    spec_common = [0,1,2,3,4,5,6,7,8,9,11,12,15,16,17,19]
    Acom = A[spec_common].T[spec_common].T
    rcom = rhealthy[100:,spec_common]
else:
    spec_common = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16]
    Acom = A[spec_common].T[spec_common].T
    rcom = rhealthy[100:,spec_common]

##### uncomment to save the information
# np.savetxt('A'+type_pat+'.csv', Acom, delimiter=",")
# np.savetxt('r'+type_pat+'.csv', rcom, delimiter=",")
