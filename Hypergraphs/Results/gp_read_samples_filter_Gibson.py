from pyeda.inter import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from functions16 import Compl, truth_table, Error_bet


Num = 16

x = exprvars('x', Num)

pp = 2**Num

####################################################

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


df_col0 = pd.read_csv('Tables_'+pat+'/All_samples.csv', usecols= ["Complejidad", "Error", "Modelo"])
compl0 = np.array(df_col0['Complejidad'])
error0 = np.array(df_col0['Error'])
model0 = np.array(df_col0['Modelo'])    

sort_compl = np.sort(compl0)
single_compl = [sort_compl[0]]
for i in np.arange(1,len(sort_compl)):
    current = sort_compl[i]
    last = sort_compl[i-1]
    if Not(current==last):
        single_compl.append(current)

single_compl = np.array(single_compl)

compl_filt = []
error_filt = []
model_filt = []  

for cx in single_compl:
    loc = np.where(compl0==cx)[0]
    if len(loc)==1:
        com = compl0[loc[0]]
        compl_filt.append(com)
        err = error0[loc[0]]
        error_filt.append(err)
        sur = model0[loc[0]]
        model_filt.append(sur)
    else:
        if cx==2:
            TYPE=[]
            for i in range(len(loc)):
                sur = model0[loc[i]]
                sur = eval(sur)
                TYPE.append(type(sur))
                loc1 = np.where(np.array(TYPE)!=int)[0]
            if len(loc1)>0:
                com = compl0[loc[loc1][0]]
                compl_filt.append(com)
                err = error0[loc[loc1][0]]
                error_filt.append(err)
                sur = model0[loc[loc1][0]]
                model_filt.append(sur)
                for k in loc[loc1][1:]:
                    if type(sur)==str:
                        sur = eval(sur)
                    sur2 = model0[k]
                    sur2 = eval(sur2)
                    if type(sur)!=int:
                        if Not(sur.equivalent(sur2)):
                            com = compl0[loc[i]]
                            compl_filt.append(com)
                            err = error0[loc[i]]
                            error_filt.append(err)
                            sur = model0[loc[i]]
                            model_filt.append(sur)
        else:
            com = compl0[loc[0]]
            compl_filt.append(com)
            err = error0[loc[0]]
            error_filt.append(err)
            sur = model0[loc[0]]
            model_filt.append(sur)
            if cx==0:
                for i in loc[1:]:
                    sur = eval(sur)
                    sur2 = eval(model0[i])
                    if Not(sur==sur2):
                        com = compl0[i]
                        compl_filt.append(com)
                        err = error0[i]
                        error_filt.append(err)
                        sur = model0[i]
                        model_filt.append(sur)
            else:
                for i in np.arange(1,len(loc)):
                    sur2 = model0[loc[i]]
                    if type(sur)==str:
                        sur = eval(sur)
                    sur2 = eval(sur2)
                    if type(sur2)==str:
                        sur = eval(sur2)
                    if Not(sur.equivalent(sur2)):
                        com = compl0[loc[i]]
                        compl_filt.append(com)
                        err = error0[loc[i]]
                        error_filt.append(err)
                        sur = model0[loc[i]]
                        model_filt.append(sur)                


with open('Tables_'+pat+'/All_samples_filt.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Complejidad", "Error", "Modelo"])
    for i in range(len(compl_filt)):
        writer.writerow([compl_filt[i],error_filt[i], model_filt[i]])
