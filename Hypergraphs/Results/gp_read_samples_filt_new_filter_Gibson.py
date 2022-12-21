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

####################################################

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

df_col0 = pd.read_csv('Tables_'+pat+'/All_samples_filt.csv', usecols= ["Complejidad", "Error", "Modelo"])
compl0 = np.array(df_col0['Complejidad'])
error0 = np.array(df_col0['Error'])
model0 = np.array(df_col0['Modelo'])    

single_compl = [compl0[0]]
for i in np.arange(1,len(compl0)):
    current = compl0[i]
    last = compl0[i-1]
    if Not(current==last):
        single_compl.append(current)

single_compl = np.array(single_compl)

compl_filt = []
error_filt = []
model_filt = []  

loc = np.where(compl0==0)[0]
for i in range(len(loc)):
    com = compl0[loc[i]]
    compl_filt.append(com)
    err = error0[loc[i]]
    error_filt.append(err)
    sur = model0[loc[i]]
    model_filt.append(sur)            
for cx in single_compl[1:]:
    loc = np.where(compl0==cx)[0]
    if len(loc)==2:
        com = compl0[loc[0]]
        compl_filt.append(com)
        err = error0[loc[0]]
        error_filt.append(err)
        sur = model0[loc[0]]
        model_filt.append(sur)        
        mod1 = eval(model0[loc[0]])
        mod2 = eval(model0[loc[1]])
        if Not(mod1.equivalent(mod2)):
            com = compl0[loc[1]]
            compl_filt.append(com)
            err = error0[loc[1]]
            error_filt.append(err)
            sur = model0[loc[1]]
            model_filt.append(sur)        
    else:
        mod_list = []
        com = compl0[loc[0]]
        compl_filt.append(com)
        err = error0[loc[0]]
        error_filt.append(err)
        sur = model0[loc[0]]
        model_filt.append(sur)
        mod1 = eval(model0[loc[0]])
        mod_list.append(sur)
        for i in np.arange(1,len(loc)):
            mod2 = model0[loc[i]]
            if Model_equi(mod_list,mod2):
                com = compl0[loc[i]]
                compl_filt.append(com)
                err = error0[loc[i]]
                error_filt.append(err)
                sur = model0[loc[i]]
                model_filt.append(sur)
                mod_list.append(sur)


with open('Tables_'+pat+'/All_samples_filt_new.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Complejidad", "Error", "Modelo"])
    for i in range(len(compl_filt)):
        writer.writerow([compl_filt[i],error_filt[i], model_filt[i]])

#################################################

