from pyeda.inter import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from functions16 import Compl, truth_table, Error_bet
from functions16 import is_atom, all_parts_atoms, atom_sub_model
from functions16 import model_apa_sub_model, model_napa_sub_model
import os


if not os.path.exists('Purple_Pareto_'+pat):
    os.makedirs('Purple_Pareto_'+pat)

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
all_compl = np.array(df_col['Complejidad'])
all_models = np.array(df_col['Modelo'])    
NN_all = len(all_models)

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

Pareto_table = pd.read_csv('Tables_'+pat+'/Pareto_Table_purple.csv', usecols= ["Complejidad", "Error","Modelo","Numero_lista"])
model_Pareto = np.array(Pareto_table['Modelo'])
ind_list = np.array(Pareto_table['Numero_lista'])

tie_list = pd.read_csv('Tables_'+pat+'/Pareto_Table_new_Tielist.csv',header = None)
tie_list = np.array(tie_list)[0]

nn = len(tie_list)
Pareto_number_list = []
Pareto_number_models = []
if nn == 1:
    aa = tie_list[0]
    if aa == 0:
        Pareto_table = pd.read_csv('Tables_'+pat+'/Pareto_Table_purple.csv', usecols= ["Complejidad", "Error","Modelo","Numero_lista"])
        model_Pareto = np.array(Pareto_table['Modelo'])
        ind_list = np.array(Pareto_table['Numero_lista'])
        NN = len(model_Pareto)
        if NN ==1:
            Pareto_number_models = [aa]
        else:
            for kk in range(NN):
                # modeli = eval(model_Pareto[kk])
                # for jj in range(128):
                #     modelj = eval(all_models[jj])
                #     if Error_bet(modeli, modelj)==0.0:
                #         Pareto_number_models.append(jj)
                indi = ind_list[kk]
                Pareto_number_models.append(indi)
        with open('Purple_Pareto_'+pat+'/Pareto_Table'+str(aa)+'_arrows.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(Pareto_number_models)
    else:
        Pareto_table = pd.read_csv('Tables_'+pat+'/Pareto_Table_purple.csv', usecols= ["Complejidad", "Error","Modelo","Numero_lista"])
        model_Pareto = np.array(Pareto_table['Modelo'])
        ind_list = np.array(Pareto_table['Numero_lista'])
        NN = len(model_Pareto)
 
        Tie_table = pd.read_csv('Tables_'+pat+'/Pareto_Table_new_Tie_'+str(aa)+'.csv', usecols= ["Complejidad", "Error","Modelo","Numero_lista"])
        model_Tie = np.array(Tie_table['Modelo'])
        ind_list_Tie = np.array(Tie_table['Numero_lista'])
        Nt = len(model_Tie)

        Tie_table_arrows = pd.read_csv('Tables_'+pat+'/Pareto_Table_new_arrows.csv', header = None)
        Tie_table_arrows = np.array(Tie_table_arrows)[0]

        loc1 = np.where(Tie_table_arrows==1)[0]
        if len(loc1)>0:
            for ind in loc1:
                for kk in range(NN):
                    if kk == aa:
                        # modeli = eval(model_Tie[ind])
                        # for jj in range(128):
                        #     modelj = eval(all_models[jj])
                        #     if Error_bet(modeli, modelj)==0.0:
                        #         Pareto_number_models.append(jj)
                        indi = ind_list_Tie[ind]
                        Pareto_number_models.append(indi)
                    else:
                        # modeli = eval(model_Pareto[kk])
                        # for jj in range(128):
                        #     modelj = eval(all_models[jj])
                        #     if Error_bet(modeli, modelj)==0.0:
                        #         Pareto_number_models.append(jj)
                        indi = ind_list[kk]
                        Pareto_number_models.append(indi)
                Pareto_number_list.append(Pareto_number_models)
                Pareto_number_models = []
        else:
            Pareto_table = pd.read_csv('Tables_'+pat+'/Pareto_Table_purple.csv', usecols= ["Complejidad", "Error","Modelo"])
            model_Pareto = np.array(Pareto_table['Modelo'])
            NN = len(model_Pareto)
            for kk in range(NN):
                # modeli = eval(model_Pareto[kk])
                # for jj in range(NN_all):
                #     modelj = eval(all_models[jj])
                #     if Error_bet(modeli, modelj)==0.0:
                #         Pareto_number_models.append(jj)
                indi = ind_list[kk]
                Pareto_number_models.append(indi)
            Pareto_number_list.append(Pareto_number_models)
            Pareto_number_models = []
        # if len(loc1)==1:
        #     for kk in range(NN):
        #         if kk == aa:
        #             modeli = eval(model_Tie[loc1[0]])
        #             for jj in range(128):
        #                 modelj = eval(all_models[jj])
        #                 if Error_bet(modeli, modelj)==0.0:
        #                     Pareto_number_models.append(jj)
        #         else:
        #             modeli = eval(model_Pareto[kk])
        #             for jj in range(128):
        #                 modelj = eval(all_models[jj])
        #                 if Error_bet(modeli, modelj)==0.0:
        #                     Pareto_number_models.append(jj)
        #     Pareto_number_list.append(Pareto_number_models)
        #     Pareto_number_models = []
        # else:
        #     for ss in range(Nt):
        #         for kk in range(NN):
        #             if kk == aa:
        #                 if Tie_table_arrows[ss]==1:
        #                     modeli = eval(model_Tie[ss])
        #                     for jj in range(128):
        #                         modelj = eval(all_models[jj])
        #                         if Error_bet(modeli, modelj)==0.0:
        #                             Pareto_number_models.append(jj)
        #             else:
        #                 modeli = eval(model_Pareto[kk])
        #                 for jj in range(128):
        #                     modelj = eval(all_models[jj])
        #                     if Error_bet(modeli, modelj)==0.0:
        #                         Pareto_number_models.append(jj)
        #         Pareto_number_list.append(Pareto_number_models)
        #         Pareto_number_models = []

        # for ss in range(Nt):
        #     for kk in range(NN):
        #         if kk == aa:
        #             if Tie_table_arrows[ss]==1:
        #                 modeli = eval(model_Tie[ss])
        #                 for jj in range(128):
        #                     modelj = eval(all_models[jj])
        #                     if Error_bet(modeli, modelj)==0.0:
        #                         Pareto_number_models.append(jj)
        #         else:
        #             modeli = eval(model_Pareto[kk])
        #             for jj in range(128):
        #                 modelj = eval(all_models[jj])
        #                 if Error_bet(modeli, modelj)==0.0:
        #                     Pareto_number_models.append(jj)
        #     Pareto_number_list.append(Pareto_number_models)
        #     Pareto_number_models = []

        with open('Purple_Pareto_'+pat+'/Pareto_Table_arrows.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for i in range(len(Pareto_number_list)):
                writer.writerow(Pareto_number_list[i])
elif nn == 2:
    Pareto_table = pd.read_csv('Tables_'+pat+'/Pareto_Table_purple.csv', usecols= ["Complejidad", "Error","Modelo","Numero_lista"])
    model_Pareto = np.array(Pareto_table['Modelo'])
    ind_list = np.array(Pareto_table['Numero_lista'])
    NN = len(model_Pareto)

    Tie_table_arrows = pd.read_csv('Tables_'+pat+'/Pareto_Table_new_arrows.csv', header = None)
    Tie_table_arrows = np.array(Tie_table_arrows)

    aa1 = tie_list[0]
    aa2 = tie_list[1]

    Tie_table1 = pd.read_csv('Tables_'+pat+'/Pareto_Table_new_Tie_'+str(aa1)+'.csv', usecols= ["Complejidad", "Error","Modelo","Numero_lista"])
    model_Tie1 = np.array(Tie_table1['Modelo'])
    ind_list_Tie1 = np.array(Tie_table1['Numero_lista'])
    Nt1 = len(model_Tie1)

    Tie_table2 = pd.read_csv('Tables_'+pat+'/Pareto_Table_new_Tie_'+str(aa2)+'.csv', usecols= ["Complejidad", "Error","Modelo","Numero_lista"])
    model_Tie2 = np.array(Tie_table2['Modelo'])
    ind_list_Tie2 = np.array(Tie_table2['Numero_lista'])
    Nt2 = len(model_Tie2)

    for kk in range(Nt2):
        model_Pareto[aa2] = model_Tie2[kk]
        ind_list[aa2] = ind_list_Tie2[kk]
        arrow = Tie_table_arrows[kk]
        for jj in range(Nt1):
            if arrow[jj]==1:
                model_Pareto[aa1] = model_Tie1[jj]
                ind_list[aa1] = ind_list_Tie1[jj]
                for kkk in range(NN):
                    indi = ind_list[kkk]
                    Pareto_number_models.append(indi)
                Pareto_number_list.append(Pareto_number_models)
                Pareto_number_models = []
    with open('Purple_Pareto_'+pat+'/Pareto_Table_arrows.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(Pareto_number_list)):
            writer.writerow(Pareto_number_list[i])

elif nn == 3:
    Pareto_table = pd.read_csv('Tables_'+pat+'/Pareto_Table_purple.csv', usecols= ["Complejidad", "Error","Modelo","Numero_lista"])
    model_Pareto = np.array(Pareto_table['Modelo'])
    ind_list = np.array(Pareto_table['Numero_lista'])
    NN = len(model_Pareto)

    Tie_table_arrows = pd.read_csv('Tables_'+pat+'/Pareto_Table_new_arrows.csv', header = None)
    Tie_table_arrows = np.array(Tie_table_arrows)

    aa1 = tie_list[0]
    aa2 = tie_list[1]
    aa3 = tie_list[2]

    Tie_table1 = pd.read_csv('Tables_'+pat+'/Pareto_Table_new_Tie_'+str(aa1)+'.csv', usecols= ["Complejidad", "Error","Modelo","Numero_lista"])
    model_Tie1 = np.array(Tie_table1['Modelo'])
    ind_list_Tie1 = np.array(Tie_table1['Numero_lista'])
    Nt1 = len(model_Tie1)

    Tie_table2 = pd.read_csv('Tables_'+pat+'/Pareto_Table_new_Tie_'+str(aa2)+'.csv', usecols= ["Complejidad", "Error","Modelo","Numero_lista"])
    model_Tie2 = np.array(Tie_table2['Modelo'])
    ind_list_Tie2 = np.array(Tie_table2['Numero_lista'])
    Nt2 = len(model_Tie2)

    Tie_table3 = pd.read_csv('Tables_'+pat+'/Pareto_Table_new_Tie_'+str(aa3)+'.csv', usecols= ["Complejidad", "Error","Modelo","Numero_lista"])
    model_Tie3 = np.array(Tie_table3['Modelo'])
    ind_list_Tie3 = np.array(Tie_table3['Numero_lista'])
    Nt3 = len(model_Tie3)


    for ss in range(Nt3):
        model_Pareto[aa3] = model_Tie3[ss]
        ind_list[aa3] = ind_list_Tie3[ss]
        arrow1 = Tie_table_arrows[ss]
        for kk in range(Nt2):
            if arrow1[kk]==1:
                model_Pareto[aa2] = model_Tie2[kk]
                ind_list[aa2] = ind_list_Tie2[kk]
                arrow = Tie_table_arrows[kk]
                for jj in range(Nt1):
                    if arrow[jj]==1:
                        model_Pareto[aa1] = model_Tie1[jj]
                        ind_list[aa1] = ind_list_Tie1[jj]
                        for kkk in range(NN):
                            indi = ind_list[kkk]
                            Pareto_number_models.append(indi)
                        Pareto_number_list.append(Pareto_number_models)
                        Pareto_number_models = []
    with open('Purple_Pareto_'+pat+'/Pareto_Table_arrows.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(Pareto_number_list)):
            writer.writerow(Pareto_number_list[i])