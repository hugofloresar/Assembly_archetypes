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

####################################################
### Generate the array of errors for each model

Complexity_arr = compl
Error_arr = []
Samples_arr = []

nn = len(compl)
for i in range(nn):
    expr = eval(model[i])
    error = Error_bet_Fc(expr)
    Error_arr.append(error)
    Samples_arr.append(expr)

Error_arr = np.array(Error_arr)
Samples_arr = np.array(Samples_arr)

###########################################################

def Compute_Pareto_Front(Complexity_arr,Error_arr,Samples_arr):
    complexity_Par = []
    error_Par = []
    model_Par = []
    number = []

    ### find minimum complexity
    cc0 = min(Complexity_arr)
    ### loc of minimum complexity
    loc_c0 = np.where(Complexity_arr==cc0)[0]
    ### min error from all models of minimum complexity
    e0 = min(Error_arr[loc_c0])
    loc_e0 = np.where(Error_arr[loc_c0]==e0)[0]
    ind = loc_c0[loc_e0][0]
    print(ind)
    expr = Samples_arr[ind]
    number.append(ind)
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
                ind = loc_ci[loc_ei][0]
                print(ind)
                expr = Samples_arr[loc_ci][loc_ei][0]
                number.append(ind)
                model_Par.append(expr)

    Compl_arr = complexity_Par
    Compl_arr = np.array(Compl_arr)
    Err_arr = error_Par
    Err_arr = np.array(Err_arr)
    Model_arr = model_Par
    Model_arr = np.array(Model_arr)

    return Compl_arr, Err_arr, Model_arr, number

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

Compl_arr, Err_arr, Model_arr, n_arr = Compute_Pareto_Front(Complexity_arr,Error_arr,Samples_arr)

with open('Tables_'+pat+'/Pareto_Table_new.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Complejidad", "Error", "Modelo","Numero_lista"])
    for i in range(len(Err_arr)):
        writer.writerow([Compl_arr[i],Err_arr[i], Model_arr[i],n_arr[i]])

NN = len(Compl_arr)

if NN>2:
    lista_ties = []
    # number = []
    for jj in np.arange(1,NN-1):
        locp = np.where(Error_arr == Err_arr[jj])[0]
        Model_locp = Samples_arr[locp]
        Compl_locp = Complexity_arr[locp]

        locp_complx = np.where(Compl_locp == Compl_arr[jj])[0]
        Model_Tiep = Model_locp[locp_complx]
        ind = locp[locp_complx]
        # number.append(ind)
        nn2 = len(locp_complx)
        if nn2>1:
            lista_ties.append(jj)
            with open('Tables_'+pat+'/Pareto_Table_new_Tie_'+str(jj)+'.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Complejidad", "Error", "Modelo","Numero_lista"])
                for i in range(nn2):
                    writer.writerow([Compl_arr[jj],Err_arr[jj],Model_Tiep[i],ind[i]])

    ll = len(lista_ties)
    if ll>0:
        with open('Tables_'+pat+'/Pareto_Table_new_Tielist.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(lista_ties)
    else:
        lista_ties = [0]
        with open('Tables_'+pat+'/Pareto_Table_new_Tielist.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(lista_ties)
else:
    lista_ties = [0]
    with open('Tables_'+pat+'/Pareto_Table_new_Tielist.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(lista_ties)

Compl_arr2 = np.copy(Compl_arr)
Err_arr2 = np.copy(Err_arr)
Model_arr2 = np.copy(Model_arr)
n_arr2 = np.copy(n_arr)

nn = len(Model_arr)
k = nn-1
while k > 0:
    if is_submodel(Model_arr[k-1],Model_arr[k]):
        k = k-1
    else:
        lower_compl = np.where(Complexity_arr<Compl_arr[k])[0]
        Complex_arr_sur = Complexity_arr[lower_compl]
        Error_arr_sur = Error_arr[lower_compl]
        Samples_arr_sur = Samples_arr[lower_compl]

        lower_error = np.where(Error_arr_sur<Err_arr[k-2])[0]
        Samples_arr_sur2 = Samples_arr_sur[lower_error]
        Complex_arr_sur2 = Complex_arr_sur[lower_error]
        Error_arr_sur2 = Error_arr_sur[lower_error]
        numb_arr_sur2 = lower_compl[lower_error]
        
        ll = len(Samples_arr_sur2)

        if ll>0:
            suma = 0
            for i in range(ll):
                if is_submodel(Samples_arr_sur2[i],Model_arr[k]):
                    Compl_arr2[k-1] = Complex_arr_sur2[i]
                    Err_arr2[k-1] = Error_arr_sur2[i]
                    Model_arr2[k-1] = Samples_arr_sur2[i]
                    n_arr2[k-1] = numb_arr_sur2[i]
                else:
                    suma = suma + 1
                    print(i)
            if suma==ll:
                Compl_arr2 = np.delete(Compl_arr2,[k-1])
                Err_arr2 = np.delete(Err_arr2,[k-1])
                Model_arr2 = np.delete(Model_arr2,[k-1])
                n_arr2 = np.delete(n_arr2,[k-1])

            k = k-1
        else:
            Compl_arr2 = np.delete(Compl_arr2,[k-1])
            Err_arr2 = np.delete(Err_arr2,[k-1])
            Model_arr2 = np.delete(Model_arr2,[k-1])
            n_arr2 = np.delete(n_arr2,[k-1])
            k = k-1


with open('Tables_'+pat+'/Pareto_Table_purple.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Complejidad", "Error", "Modelo","Numero_lista"])
    for i in range(len(Err_arr2)):
        writer.writerow([Compl_arr2[i],Err_arr2[i], Model_arr2[i],n_arr2[i]])
