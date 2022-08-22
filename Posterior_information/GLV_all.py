import numpy as np
import scipy as sp
import pylab as pl
import sys
import os
from scipy import integrate, optimize


def rhs_mono(x, t, p):
    fx = np.zeros(1)
    fx[0] = x[0]*p[0] + p[1]*x[0]**2
    return fx

def rhs_pair(x, t, p):
    fx = np.zeros(2)
    fx[0] = x[0]*p[0][0] + p[0][1]*x[0]**2 + p[0][2]*x[0]*x[1]
    fx[1] = x[1]*p[1][0] + p[1][1]*x[0]*x[1] + p[1][2]*x[1]**2
    return fx

def rhs_trio(x, t, p):
    fx = np.zeros(3)
    fx[0] = x[0]*p[0][0] + p[0][1]*x[0]**2 + p[0][2]*x[0]*x[1] + p[0][3]*x[0]*x[2]
    fx[1] = x[1]*p[1][0] + p[1][1]*x[0]*x[1] + p[1][2]*x[1]**2 + p[1][3]*x[1]*x[2]
    fx[2] = x[2]*p[2][0] + p[2][1]*x[0]*x[2] + p[2][2]*x[1]*x[2] + p[2][3]*x[2]**2
    return fx

def rhs_four(x, t, p):
    fx = np.zeros(4)
    for i in range(4):
        suma = p[i][0]
        for j in np.arange(1,5):
            suma = suma + p[i][j] * x[j-1]
        fx[i] = x[i]*suma
    return fx

def rhs_seven(x, t, p):
    fx = np.zeros(7)
    for i in range(7):
        suma = p[i][0]
        for j in np.arange(1,8):
            suma = suma + p[i][j] * x[j-1]
        fx[i] = x[i]*suma
    return fx

def rhs_eight(x, t, p):
    fx = np.zeros(8)
    for i in range(8):
        suma = p[i][0]
        for j in np.arange(1,9):
            suma = suma + p[i][j] * x[j-1]
        fx[i] = x[i]*suma
    return fx
