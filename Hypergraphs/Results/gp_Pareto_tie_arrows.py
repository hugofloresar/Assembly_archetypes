from pyeda.inter import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from functions16 import Compl, truth_table, Error_bet
from functions16 import is_atom, all_parts_atoms, atom_sub_model
from functions16 import model_apa_sub_model, model_napa_sub_model
import os

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

df_col = pd.read_csv('All_models.csv', usecols= ["Complejidad", "Modelo"])
compl = np.array(df_col['Complejidad'])
model = np.array(df_col['Modelo'])    

####################################################

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

df_col = pd.read_csv('Gibson_full/Gibson'+pat+'_simplify.csv', usecols= ["Modelo"])
F = df_col['Modelo'][0]
Fc = eval(F)

###########################################################

def Compute_Pareto_Front(Complexity_arr,Error_arr,Samples_arr):
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

    Compl_arr = complexity_Par
    Compl_arr = np.array(Compl_arr)
    Err_arr = error_Par
    Err_arr = np.array(Err_arr)
    Model_arr = model_Par
    Model_arr = np.array(Model_arr)

    return Compl_arr, Err_arr, Model_arr

def is_submodel(modeli, modelj):
    if type(modeli)==int:
        return 1
    else:
        if type(modelj)!=int:
            if is_atom(modeli):
                if atom_sub_model(modeli, modelj):
                    return 1
                else:
                    return 0
            else:
                if all_parts_atoms(modeli):
                    if model_apa_sub_model(modeli, modelj):
                        return 1
                    else:
                        return 0
                else:
                    if model_napa_sub_model(modeli,modelj):
                        return 1
                    else:
                        return 0

################################################################

Pareto_table = pd.read_csv('Tables_'+pat+'/Pareto_Table_purple.csv', usecols= ["Complejidad", "Error","Modelo"])
model_Pareto = np.array(Pareto_table['Modelo'])    

tie_list = pd.read_csv('Tables_'+pat+'/Pareto_Table_new_Tielist.csv',header = None)
tie_list = np.array(tie_list)[0]

nn = len(tie_list)
if nn == 1:
    aa = tie_list[0]
    if aa!= 0:
        Tie1 = pd.read_csv('Tables_'+pat+'/Pareto_Table_new_Tie_'+str(aa)+'.csv', usecols= ["Complejidad", "Modelo"])
        modelTie1 = np.array(Tie1['Modelo'])    
        nn1 = len(modelTie1)
        modelP = eval(model_Pareto[aa+1])
        list_sub=[]
        for ii in range(nn1):
            modeli = eval(modelTie1[ii])
            list_sub.append(is_submodel(modeli, modelP))
        with open('Tables_'+pat+'/Pareto_Table_new_arrows.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(list_sub)
    else:
        pass
elif nn == 2:
    aa = tie_list[0]
    bb = tie_list[1]
    Tie1 = pd.read_csv('Tables_'+pat+'/Pareto_Table_new_Tie_'+str(aa)+'.csv', usecols= ["Complejidad", "Modelo"])
    modelTie1 = np.array(Tie1['Modelo'])    
    nn1 = len(modelTie1)

    Tie2 = pd.read_csv('Tables_'+pat+'/Pareto_Table_new_Tie_'+str(bb)+'.csv', usecols= ["Complejidad", "Modelo"])
    modelTie2 = np.array(Tie2['Modelo'])    
    nn2 = len(modelTie2)

    submodels = []
    for jj in range(nn2):
        print('jj= ',jj)
        modelP = eval(modelTie2[jj])
        list_sub=[]
        for ii in range(nn1):
            modeli = eval(modelTie1[ii])
            list_sub.append(is_submodel(modeli, modelP))
        submodels.append(list_sub)

    pp = len(submodels)
    with open('Tables_'+pat+'/Pareto_Table_new_arrows.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(pp):
            writer.writerow(submodels[i])

    ### missing code for values of nn bigger than 2
