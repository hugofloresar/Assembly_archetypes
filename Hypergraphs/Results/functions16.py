import numpy as np
from pyeda.inter import *
import csv
import random
import os

Num = 16

x = exprvars('x', Num)

pp = 2**Num

def is_atom(expr):
    if expr == 0:
        return True
    if expr == 1:
        return True
    else:
        compl = Compl(expr)
        if compl==1 :
            return True
        elif compl==2:
            return True
        else:
            return False

def all_parts_atoms(expr):
    rr = []
    for subm in expr.xs:
        rr.append(is_atom(subm))
    return all(rr)


def simplify_princ(Fc):
    if is_atom(Fc):
        return Fc
    elif all_parts_atoms(Fc):
        return Fc
    else:
        tt = len(Fc.xs)
        parts = farray(Fc.xs)
        if tt==0:
            sur_exp = 0
        elif tt==1:
            sur_exp = Fc
        elif tt==2:
            sur_exp = Or(parts[0],parts[1])
            sur_exp = sur_exp.to_cnf()
        else:
            sur_exp = Or(parts[0],parts[1])
            for j in np.arange(2,tt):
                sur_exp = Or(sur_exp, parts[int(j)])
                sur_exp = sur_exp.to_cnf()
        return sur_exp


def Which_variables(expr):
    if expr == 0:
        return 0
    elif expr == 1:
        return x
    else:
        if expr.depth==0:
            tru = np.zeros(Num,dtype=int)
            for i in range(Num):
                tru[i] = (expr == x[i]) or (expr == ~x[i])
            return tru
        elif expr.depth==1:
            dd = len(expr.xs)
            var_list = np.zeros(Num,dtype=int)
            for j in range(dd):
                which = Which_variables(expr.xs[j])
                loc = np.where(which==1)[0]
                var_list[loc]=1
            return var_list
        else:
            dd = len(expr.xs)
            var_list = np.zeros(Num,dtype=int)
            for j in range(dd):
                which = Which_variables(expr.xs[j])
                loc = np.where(which==1)[0]
                var_list[loc]=1
            return var_list


def Compl(expr):
    if expr == 0:
        return 0
    if expr == 1:
        return 0
    else:
        if expr.depth==0:
            tru = np.zeros(Num,dtype=int)
            for i in range(Num):
                tru[i] = expr == x[i]
            if any(tru):
                return 1
            else:
                return 2
        elif expr.depth==1:
            nn = len(expr.xs)
            tru = np.zeros(Num,dtype=int)
            for i in range(nn):
                tru[i] = Compl(expr.xs[i])
            loc = np.where(tru==2)[0]
            return len(loc) + np.sum(Which_variables(expr)) + 1
        else:
            nn = len(expr.xs)
            complexity = 0
            for i in range(nn):
                submodela = expr.xs[i]
                complexity = complexity + (Compl(submodela)-np.sum(Which_variables(submodela)))
            return complexity + np.sum(Which_variables(expr)) + 1                


def truth_table(expr):
    tt_values = np.zeros(pp-1,dtype=int)
    if expr==0:
        return tt_values
    elif expr==1:
        return np.ones(pp-1,dtype=int)
    else:
        list_point = list(iter_points([x[0],x[1],x[2],x[3],x[4],
                            x[5],x[6],x[7],x[8],x[9],x[10],x[11],
                            x[12],x[13],x[14],x[15]
                                      ]))
        for k in np.arange(1,pp):
            point = list_point[k]
            tt_values[k-1] = expr.restrict(point)
        return tt_values


def Error_bet(expr1,expr2):
    tt1 = truth_table(expr1)
    if expr2 ==1:
        tt2 = np.ones(pp-1,dtype=int)
    if expr2 ==0:
        tt2 = np.zeros(pp-1,dtype=int)
    else: 
        tt2 = truth_table(expr2)
    Error = np.sum(abs(tt1-tt2))/(pp-1)
    return Error

def atom_sub_model(atom, model):
    if is_atom(model):
        if (atom==1 or atom==0):
            if (model==1 or model==0):
                return atom==model
            else:
                return False                
        elif atom.equivalent(model):
            return True
        else:
            return False
    else:
        for subm in model.xs:
            if is_atom(subm):
                if atom_sub_model(atom, subm):
                    return True
            else:
                for subsub in subm.xs:
                    if is_atom(subsub):
                        if atom_sub_model(atom, subm):
                            return True


def model_apa_sub_model(model_apa, model):
    if is_atom(model):
        return False
    else:
        if all_parts_atoms(model_apa):
            rr = []
            for atom in model_apa.xs:
                rr.append(atom_sub_model(atom, model))
            return all(rr)
        else:
            rr = []
            if none_parts_atoms(model):
                for part in model.xs:
                    rr.append(model_apa_sub_model(model_apa,part))
                return any(rr)
            else:
                rr = []
                for atom in model_apa.xs:
                    rr.append(atom_sub_model(atom, model))
                return all(rr)


def none_parts_atoms(expr):
    rr = []
    for subm in expr.xs:
        rr.append( not is_atom(subm))
    return all(rr)

def part_submodel(part,model):
    nn = len(model.xs)
    for i in range(nn):
        expr = model.xs[i]
        if part.equivalent(expr):
            return True
    return False


def model_napa_sub_model(model_napa, model):
    if is_atom(model):
        return False
    else:
        if all_parts_atoms(model):
            return False
        else:
            if len(model_napa.xs)>len(model.xs):
                return False
            else:
                rr = []
                for part in model_napa.xs:
                    if is_atom(part):
                        rr.append(atom_sub_model(part, model))
                    else:
                        aa = model_apa_sub_model(part, model) or part_submodel(part, model)
                        # rr.append(model_apa_sub_model(part, model))
                        # rr.append(part_submodel(part, model))
                        rr.append(aa)
                return all(rr)


