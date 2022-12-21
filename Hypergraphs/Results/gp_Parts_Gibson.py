from pyeda.inter import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import random
import os
from functions16 import Compl, truth_table, Error_bet



if not os.path.exists('Tables_'+pat):
    os.makedirs('Tables_'+pat)
    
Num = 16

x = exprvars('x', Num)

TTFc = pd.read_csv('Gibson_full/Fc_truth_table_'+pat+'.csv',header = None)
TTFc = np.array(TTFc,dtype=int).T[0]

def Error_bet_Fc(expr):
    tt1 = TTFc
    if expr ==1:
        tt2 = np.ones(pp-1,dtype=int)
    if expr ==0:
        tt2 = np.zeros(pp-1,dtype=int)
    else: 
        tt2 = truth_table(expr)
    Error = np.sum(abs(tt1-tt2))/(pp-1)
    return Error

def sample_minusk(Fc,k):
    tt = len(Fc.xs)
    branches = farray(Fc.xs)
    sample = np.ones(tt,dtype=int)
    ii = random.sample(range(tt),k=k)
    sample[ii] = 0
    loc = np.where(sample==1)[0]
    ll = len(loc)
    if ll==0:
        sur_exp = 0
    elif ll==1:
        sur_exp = branches[int(loc[0])]
    elif ll==2:
        sur_exp = And(branches[int(loc[0])],branches[int(loc[1])])
    else:
        sur_exp = And(branches[int(loc[0])],branches[int(loc[1])])
        for j in np.arange(2,ll):
            sur_exp = And(sur_exp,branches[int(loc[j])])

    return sur_exp, sample


def sample_branch(Fc):
    if (Compl(Fc)==1 or Compl(Fc)==2):
        return Fc, [1]
    else:
        tt = len(Fc.xs)
        branches = farray(Fc.xs)
        sample = np.random.binomial(1,0.5,tt)
        while Not(any(sample)):
            sample = np.random.binomial(1,0.5,tt)
        loc = np.where(sample==1)[0]
        ll = len(loc)
        if ll==0:
            sur_exp = 0
        elif ll==1:
            sur_exp = branches[int(loc[0])]
        elif ll==2:
            sur_exp = Or(branches[int(loc[0])],branches[int(loc[1])])
        else:
            sur_exp = Or(branches[int(loc[0])],branches[int(loc[1])])
            for j in np.arange(2,ll):
                sur_exp = Or(sur_exp,branches[int(loc[j])])
        return sur_exp, sample


def sample_tree(Fc,k):
    exp, local = sample_minusk(Fc,k)
    loc = np.where(local==1)[0]
    ll = len(loc)
    if ll==0:
        sur_exp = 0
        return sur_exp, local, []
    elif ll==1:
        exp_sus = Fc.xs[int(loc[0])]
        sur_exp,loc1 = sample_branch(exp_sus)
        return sur_exp, local, loc1
    elif ll==2:
        sur_exp1,loc1 = sample_branch(Fc.xs[int(loc[0])])
        sur_exp2,loc2 = sample_branch(Fc.xs[int(loc[1])])
        sur_exp = And(sur_exp1,sur_exp2)
        return sur_exp, local, [loc1,loc2]
    else:
        sur_exp1,loc1 = sample_branch(Fc.xs[int(loc[0])])
        sur_exp2,loc2 = sample_branch(Fc.xs[int(loc[1])])
        sur_exp = And(sur_exp1,sur_exp2)
        loc_all = [loc1,loc2]
        for j in np.arange(2,ll):
            sur_expj,locj = sample_branch(Fc.xs[int(loc[j])])
            sur_exp = And(sur_exp,sur_expj)
            loc_all.append(locj)
        return sur_exp, local, loc_all


df_col = pd.read_csv('Gibson_full/Gibson'+pat+'_simplify.csv', usecols= ["Modelo"])
F = df_col['Modelo'][0]

Fc = eval(F)

Parts = farray(Fc.xs)

pp = 2**Num    

Samples_arrb = []
Complexity_arrb = []
Error_arrb = []

surrogate_model = 1
Samples_arrb.append(surrogate_model)
complexity = 0
Complexity_arrb.append(complexity)
error = Error_bet_Fc(surrogate_model)
Error_arrb.append(error)

surrogate_model = 0
Samples_arrb.append(surrogate_model)
complexity = 0
Complexity_arrb.append(complexity)
error = Error_bet_Fc(surrogate_model)
Error_arrb.append(error)

Parts_arr30 = []
locParts_arr30 = []
ComplexityParts_arr30 = []
ErrorParts_arr30 = []
Nsamples = 1
# Nsamples = 20
   
len_parts= len(Fc.xs)
quant = np.arange(len_parts)
j = 0
while j < len(quant)-1:
    for k in range(Nsamples):
        kk = random.randrange(quant[j],quant[j+1]+1)
        surrogate_model, sample = sample_minusk(Fc,kk)
        if surrogate_model == 0:
            print(surrogate_model)
        else:
            Parts_arr30.append(surrogate_model)
            complexity = Compl(surrogate_model)
            ComplexityParts_arr30.append(complexity)
            error = Error_bet_Fc(surrogate_model)
            ErrorParts_arr30.append(error)
            for jj in range(5):
                surrogate_model, sample, gg = sample_tree(Fc,kk)
                if surrogate_model == 0:
                    print(surrogate_model)
                else:
                    model_simpl = surrogate_model.to_cnf()
                    Parts_arr30.append(model_simpl)
                    complexity = Compl(model_simpl)
                    ComplexityParts_arr30.append(complexity)
                    error = Error_bet_Fc(model_simpl)
                    ErrorParts_arr30.append(error)
    j = j+1
        

AAA = np.concatenate((np.array(ComplexityParts_arr30), np.array(Complexity_arrb)))
BBB = np.concatenate((np.array(ErrorParts_arr30), np.array(Error_arrb)))
DDD = Parts_arr30
DDD = np.append(DDD,[Samples_arrb[0]])
DDD = np.append(DDD,[Samples_arrb[1]])

Complexity_arr = AAA 
Error_arr = BBB 
Samples_arr = DDD

with open('Tables_'+pat+'/All_samples.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Complejidad", "Error", "Modelo"])
    for i in range(len(DDD)):
        writer.writerow([AAA[i],BBB[i], DDD[i]])

# ###################################################

#### Compute Pareto Front

complexity_Par = []
error_Par = []
model_Par = []

### find minimum complexity
cc0 = min(Complexity_arr)
### loc of minimum complexity
loc_c0 = np.where(Complexity_arr==cc0)[0]
### min error from all models of minimum complexity
e0 = min(Error_arr[loc_c0])
loc_e0 = np.where(Error_arr[loc_c0]==e0)[0]
expr = Samples_arr[loc_c0[loc_e0][0]]
complexity_Par.append(0)
error_Par.append(e0)
model_Par = [expr]
NN = round(e0*(pp-1))
for i in np.arange(1,max(Complexity_arr)+1):
    loc_ci = np.where(Complexity_arr==i)[0]
    if len(loc_ci)>0:
        errori = min(Error_arr[loc_ci])
        if errori < error_Par[-1]:
            loc_ei = np.where(Error_arr[loc_ci]==errori)[0]
            complexity_Par.append(i)
            error_Par.append(errori)
            expr = Samples_arr[loc_ci][loc_ei][0]
            model_Par.append(expr)
    
###############################################
Compl_arr = complexity_Par
Compl_arr = np.array(Compl_arr)
Err_arr = error_Par
Err_arr = np.array(Err_arr)
Model_arr = model_Par
Model_arr = np.array(Model_arr)

with open('Tables_'+pat+'/Pareto_Table.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Complejidad", "Error", "Modelo"])
    for i in range(len(Err_arr)):
        writer.writerow([Compl_arr[i],Err_arr[i], Model_arr[i]])

