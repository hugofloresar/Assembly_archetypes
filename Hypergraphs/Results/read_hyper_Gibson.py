from pyeda.inter import *
import numpy as np
import pandas as pd
import csv


Num = 16
x = exprvars('x', Num)

pat = 'h'
# pat = 'uc'

def simplify_princ(Fc):
    tt = len(Fc.xs)
    parts = farray(Fc.xs)
    if tt==0:
        sur_exp = 0
    elif tt==1:
        sur_exp = Fc
    elif tt==2:
        sur_exp = Or(parts[0],parts[1])
    else:
        sur_exp = Or(parts[0],parts[1])
        for j in np.arange(2,tt):
            sur_exp = Or(sur_exp, parts[int(j)])
            sur_exp = sur_exp.to_cnf() 

    return sur_exp


data_file_delimiter = ','
largest_column_count = 0

# Loop the data lines
with open('Gibson_full/H'+str(pat)+'.dat', 'r') as temp_f:
    # Read the lines
    lines = temp_f.readlines()

    for l in lines:
        # Count the column count for the current line
        column_count = len(l.split(data_file_delimiter)) + 1

        # Set the new most column count
        largest_column_count = column_count if largest_column_count < column_count else largest_column_count

# Close file
temp_f.close()
    
    
number = len(lines)
hyper = []
for i in range(number):
    AA = np.zeros(Num, dtype = int)
    cand = lines[i].split('\n')[0]
    if len(cand)==1:
        a = eval(cand)
        AA[a-1] = 1
        hyper.append(AA)
    else:
        # a = np.array(cand.split('\t'), dtype = int)
        a = np.array(cand.split(','), dtype = int)
        for j in range(len(a)):
            b = a[j]
            AA[b-1] = 1
        hyper.append(AA)


TT = np.zeros((2**Num,Num+1),dtype = int)
for i in range(2**Num):
    TT[i][:Num] = uint2bdds(i,Num)


TT1 = TT[:,:Num]

for i in range(len(hyper)):
    cc = hyper[i]
    for j in range(len(TT1)):
        if (cc == TT1[j]).all():
            TT[:,Num][j]=1

a = str(TT[:,Num][0])
for i in range(len(TT)-1):
    b =  str(TT[:,Num][i+1])
    a = a+b

with open('Gibson_full/Fc_truth_table_'+pat+'.csv.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(TT[:,-1][1:])):
        writer.writerow([TT[:,-1][1:][i]])

f = truthtable(x, a)

F = truthtable2expr(f)

FF = simplify_princ(F)

with open('Gibson_full/Gibson'+str(pat)+'_simplify.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Modelo"])
    writer.writerow([FF])    

