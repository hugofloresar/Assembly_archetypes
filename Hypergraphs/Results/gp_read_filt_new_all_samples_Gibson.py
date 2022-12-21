from pyeda.inter import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from functions16 import Compl, truth_table, Error_bet

Num = 16

x = exprvars('x', Num)

pp = 2**Num


def Model_equi(model_list,expr):
    nn = len(model_list)
    expr = eval(expr)
    gg = []
    for i in range(nn):
        mod = eval(model_list[i])
        gg.append(Not(mod.equivalent(expr)))
    suma = np.sum(np.array(gg,dtype=int))
    return suma==nn


###################################################

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

pat = 'h'

df_col0 = pd.read_csv('Tables_'+pat+'/All_samples_filt_new.csv', usecols= ["Complejidad", "Error", "Modelo"])
compl0 = np.array(df_col0['Complejidad'])
error0 = np.array(df_col0['Error'])
model0 = np.array(df_col0['Modelo'])    

all_models = []
all_compls = [] 
for k in range(len(model0)):
    all_models.append(eval(model0[k]))
    all_compls.append(compl0[k]) 

for pat in ['uc']:
    df_col1 = pd.read_csv('Tables_'+pat+'/All_samples_filt_new.csv', usecols= ["Complejidad", "Error", "Modelo"])
    compl1 = np.array(df_col1['Complejidad'])
    error1 = np.array(df_col1['Error'])
    model1 = np.array(df_col1['Modelo'])    

    if len(compl1)>1:
    
        for i in range(len(compl1)):
            loc = np.where(np.array(all_compls) == compl1[i])[0]
            if len(loc) == 0:
                surrogate = eval(model1[i])
                all_models.append(surrogate)
                all_compls.append(compl1[i])
            elif len(loc) == 1:
                surrogate = eval(model1[i])
                if compl1[i]==0:
                    current = all_models[loc[0]]
                    # if Not(surrogate-current == 0):
                    #     all_models.append(surrogate)
                    #     all_compls.append(compl1[i])
                else:
                    current = all_models[loc[0]]
                    if Not(current.equivalent(surrogate)):
                        all_models.append(surrogate)
                        all_compls.append(compl1[i])
            else:
                surrogate = eval(model1[i])
                if compl1[i]==0:
                    err = []
                    for k in range(len(loc)):
                        current = all_models[loc[k]]
                        err.append(surrogate-current)
                    err = np.array(err)
                    if Not(any(err==0)):
                        all_models.append(surrogate)
                        all_compls.append(compl1[i])
                else:
                    current = all_models[loc[0]]
                    rr = []
                    for k in range(len(loc)):
                        current = all_models[loc[k]]
                        if type(current)!=int:
                            rr.append(current.equivalent(surrogate))
                    if Not(any(rr)):
                        all_models.append(surrogate)
                        all_compls.append(compl1[i])                                        


nn = len(all_compls)

with open('All_models.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Complejidad", "Modelo"])
    for i in range(nn):
        writer.writerow([all_compls[i],all_models[i]])

########################################################
