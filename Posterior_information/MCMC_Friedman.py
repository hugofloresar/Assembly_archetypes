### import libraries
from GLV_all import rhs_mono, rhs_pair, rhs_trio, rhs_seven, rhs_eight
import numpy as np
import pylab as pl
import pytwalk
from csv import reader
import os
from tempfile import TemporaryFile
from scipy import integrate

if not os.path.exists('GLV/chain'):
    os.makedirs('GLV/chain')

### list species and time observation
list_sp = ['Ea', 'Pa', 'Pch', 'Pci', 'Pf', 'Pp' ,'Pv', 'Sm']
t_data_all = np.array([0.0,40.0,80.0,120.0,160.0,200.0])

##################################################################################
'''Loading monoculture data, with 64 experiments'''

### two esp by species
#exp_mono = [3,5,9,13,18,20,25,29,35,38,44,46,50,56,60,62]
exp_mono = np.arange(1,65)

with open('Friedman_data/data_species_mono.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    list_of_rows = list(csv_reader)

exp_mono_all = []
for i in range(len(exp_mono)):
    value_mono = list_of_rows[exp_mono[i]]
    exp_mono_all.append(value_mono)

def fromlist_to_array_mono(value):
    species_vec, t_data_vec, noisy_data_all = value
    species_vec_mono = eval(species_vec)
    species_vec_mono = np.array(species_vec_mono)
    t_data_vec = eval(t_data_vec)
    noisy_data_all = eval(noisy_data_all)

    t_data_mono = []
    noisy_data_mono = []

    for i in range(6):
        if t_data_vec[i]==1:
            t_data_mono.append(t_data_all[i])
            noisy_data_mono.append(noisy_data_all[i])

    return species_vec_mono, t_data_mono, noisy_data_mono


def eval_ode_sol_mono(value,p):
    species_vec_mono, t_data_mono, noisy_data_mono = value
    p_mono = p_data(species_vec_mono, p)
    p_mono = p_mono[0]
    x0_mono = np.array([noisy_data_mono[0]])
    my_soln_mono = soln_mono(p_mono, t_data_mono, x0_mono)

    return my_soln_mono

##################################################################################
'''Loading data for cultures by pairs , with 98 experiments'''

#exp_pair = [1,2,3,5,7,9,12,14,16,18,20,22,24,25,27,28,29,34,37,39,41,43,46,48,50,53,56,58,60,\
#            61,63,65,68,70,72,74,75,78,80,82,84,86,88,89,91,93,95,97]
exp_pair = np.arange(1,99)

with open('Friedman_data/data_species_pair.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    list_of_rows = list(csv_reader)

exp_pair_all = []
for i in range(len(exp_pair)):
    value_pair = list_of_rows[exp_pair[i]]
    exp_pair_all.append(value_pair)


def fromlist_to_array_pair(value):
    species_vec, t_data_vec, noisy_data_all = value
    species_vec_pair = eval(species_vec)
    species_vec_pair = np.array(species_vec_pair)
    t_data_vec = eval(t_data_vec)
    noisy_data_all= eval(noisy_data_all)
    noisy_data_all = np.array(noisy_data_all)

    t_data_pair = []
    noisy_data_pair = np.empty((2,np.sum(t_data_vec)))

    j=0
    for i in range(6):
        if t_data_vec[i]==1:
            t_data_pair.append(t_data_all[i])
            noisy_data_pair[:,j] = noisy_data_all[:,i]
            j = j+1

    return species_vec_pair, t_data_pair, noisy_data_pair

def eval_ode_sol_pair(value,p):
    species_vec_pair, t_data_pair, noisy_data_pair = value
    p_pair = p_data(species_vec_pair, p)
    x0_pair = noisy_data_pair[:,0]
    my_soln_pair = soln_pair(p_pair, t_data_pair, x0_pair)

    return my_soln_pair


##################################################################################
'''Loading data for cultures by trio, with 298 experiments'''

#exp_trio = [3,10,18,35,57,69,89,113,149,251]
exp_trio = np.arange(1,299)

with open('Friedman_data/data_species_trio.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    list_of_rows = list(csv_reader)

exp_trio_all = []
for i in range(len(exp_trio)):
    value_trio = list_of_rows[exp_trio[i]]
    exp_trio_all.append(value_trio)


def fromlist_to_array_trio(value):
    species_vec, t_data_vec, noisy_data_all = value
    species_vec_trio = eval(species_vec)
    species_vec_trio = np.array(species_vec_trio)
    t_data_vec = eval(t_data_vec)
    noisy_data_all= eval(noisy_data_all)
    noisy_data_all = np.array(noisy_data_all)

    t_data_trio = []
    noisy_data_trio = np.empty((3,np.sum(t_data_vec)))

    j=0
    for i in range(6):
        if t_data_vec[i]==1:
            t_data_trio.append(t_data_all[i])
            noisy_data_trio[:,j] = noisy_data_all[:,j]
            j = j+1

    return species_vec_trio, t_data_trio, noisy_data_trio

def eval_ode_sol_trio(value,p):
    species_vec_trio, t_data_trio, noisy_data_trio = value
    p_trio = p_data(species_vec_trio, p)
    x0_trio = noisy_data_trio[:,0]
    my_soln_trio = soln_trio(p_trio, t_data_trio, x0_trio)

    return my_soln_trio


##################################################################################
'''Loading data for cultures by eights, from 1 to 16 experiments
and for sevenths  from 17 to 128'''

exp_eight = np.arange(1,16)
#exp_seven = [17,19,21,25,29,33,37,41,45,49,53,57,61,65,69,81,93]
exp_seven = np.arange(17,129)
with open('Friedman_data/data_species_7and8.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    list_of_rows = list(csv_reader)

exp_seven_all = []
for i in range(len(exp_seven)):
    value_seven = list_of_rows[exp_seven[i]]
    exp_seven_all.append(value_seven)

exp_eight_all = []
for i in range(len(exp_eight)):
    value_eight = list_of_rows[exp_eight[i]]
    exp_eight_all.append(value_eight)


def fromlist_to_array_seven(value):
    species_vec, t_data_vec, noisy_data_all = value
    species_vec_seven = eval(species_vec)
    species_vec_seven = np.array(species_vec_seven)
    t_data_vec = eval(t_data_vec)
    noisy_data_all= eval(noisy_data_all)
    noisy_data_all = np.array(noisy_data_all)

    t_data_seven = []
    noisy_data_seven = np.empty((7,np.sum(t_data_vec)))

    j=0
    for i in range(6):
        if t_data_vec[i]==1:
            t_data_seven.append(t_data_all[i])
            noisy_data_seven[:,j] = noisy_data_all[:,j]
            j = j+1

    return species_vec_seven, t_data_seven, noisy_data_seven

def eval_ode_sol_seven(value,p):
    species_vec_seven, t_data_seven, noisy_data_seven = value
    p_seven = p_data(species_vec_seven, p)
    x0_seven = noisy_data_seven[:,0]
    my_soln_seven = soln_seven(p_seven, t_data_seven, x0_seven)

    return my_soln_seven


def fromlist_to_array_eight (value):
    species_vec, t_data_vec, noisy_data_all = value
    species_vec_eight  = eval(species_vec)
    species_vec_eight  = np.array(species_vec_eight)
    t_data_vec = eval(t_data_vec)
    noisy_data_all= eval(noisy_data_all)
    noisy_data_all = np.array(noisy_data_all)

    t_data_eight  = []
    noisy_data_eight  = np.empty((8,np.sum(t_data_vec)))

    j=0
    for i in range(6):
        if t_data_vec[i]==1:
            t_data_eight .append(t_data_all[i])
            noisy_data_eight [:,j] = noisy_data_all[:,j]
            j = j+1

    return species_vec_eight , t_data_eight , noisy_data_eight

def eval_ode_sol_eight(value,p):
    species_vec_eight, t_data_eight, noisy_data_eight = value
    x0_eight = noisy_data_eight[:,0]
    my_soln_eight = soln_eight(p, t_data_eight, x0_eight)

    return my_soln_eight

##################################################################################

t_data = np.array([0.0,200.0])
time2 = np.linspace(0.0,200,101)

def p_data(species_vec, p):
    loc = np.where(species_vec ==1)[0]
    pp = p[loc]
    loc1 = np.concatenate(([0],loc+1))
    pp = pp.T[loc1].T
    return pp

def soln_mono(p, t_data, x0):
    return integrate.odeint(rhs_mono,x0,t_data,args=(p,))

def soln_pair(p, t_data, x0):
    return integrate.odeint(rhs_pair,x0,t_data,args=(p,))

def soln_trio(p, t_data, x0):
    return integrate.odeint(rhs_trio,x0,t_data,args=(p,))

def soln_seven(p, t_data, x0):
    return integrate.odeint(rhs_seven,x0,t_data,args=(p,))

def soln_eight(p, t_data, x0):
    return integrate.odeint(rhs_eight,x0,t_data,args=(p,))


# gamma distribution for parameters
### for the parameters
## growth rate
mu_0 = 0.2
sigma0_2 = 0.1**2
sigma0 = 0.1

## diag
mu_1 = -1.0
sigma1_2 = 0.5**2
sigma1 = 0.5
#sigma1 = 0.3

## interactions
mu_2 = 0.0
sigma2_2 = 1.0**2
sigma2 = 1.0

## likelihood variances
var=0.005**2
var1=0.01**2
sigma_2 = sigma2_2


m = 8
p = np.zeros((m,m+1))

sigma_vec0 = np.array([sigma0,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1])
sigma_vec1 = np.array([sigma0,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1])
sigma_vec2 = np.array([sigma0,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1])
sigma_vec3 = np.array([sigma0,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1])
sigma_vec4 = np.array([sigma0,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1])
sigma_vec5 = np.array([sigma0,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1])
sigma_vec6 = np.array([sigma0,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1])
sigma_vec7 = np.array([sigma0,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1,sigma1])

def energy(q): # -log of the posterior
    p[0] = q[:9]
    p[1] = q[9:18]
    p[2] = q[18:27]
    p[3] = q[27:36]
    p[4] = q[36:45]
    p[5] = q[45:54]
    p[6] = q[54:63]
    p[7] = q[63:72]


    #### for mono data
    log_like_mono = 0

    for value in exp_mono_all:
        value_mono = fromlist_to_array_mono(value)
        species_vec_mono, t_data_mono, noisy_data_mono = value_mono
        my_soln_mono = eval_ode_sol_mono(value_mono,p)
        l_like_mono = -0.5*(np.linalg.norm(noisy_data_mono - my_soln_mono))**2/var # Gaussian
        log_like_mono = log_like_mono + l_like_mono

    #### for pair data
    log_like_pair = 0

    for value in exp_pair_all:
        value_pair = fromlist_to_array_pair(value)
        species_vec_pair, t_data_pair, noisy_data_pair = value_pair
        my_soln_pair = eval_ode_sol_pair(value_pair,p)
        l_like_pair = -0.5*(np.linalg.norm(noisy_data_pair - my_soln_pair.T))**2/var # Gaussian
        log_like_pair = log_like_pair + l_like_pair

    #### for trio data
    log_like_trio = 0

    for value in exp_trio_all:
        value_trio = fromlist_to_array_trio(value)
        species_vec_trio, t_data_trio, noisy_data_trio = value_trio
        my_soln_trio = eval_ode_sol_trio(value_trio,p)
        l_like_trio = -0.5*(np.linalg.norm(noisy_data_trio - my_soln_trio.T))**2/var # Gaussian
        log_like_trio = log_like_trio + l_like_trio


    #### for seven data
    log_like_seven = 0

    for value in exp_seven_all:
        value_seven = fromlist_to_array_seven(value)
        species_vec_seven, t_data_seven, noisy_data_seven = value_seven
        my_soln_seven = eval_ode_sol_seven(value_seven,p)
        l_like_seven = -0.5*(np.linalg.norm(noisy_data_seven - my_soln_seven.T))**2/var # Gaussian
        log_like_seven = log_like_seven + l_like_seven

    #### for eigth data
    log_like_eight = 0

    for value in exp_eight_all:
        value_eight = fromlist_to_array_eight(value)
        species_vec_eight, t_data_eight, noisy_data_eight = value_eight
        my_soln_eight = eval_ode_sol_eight(value_eight,p)
        l_like_eight = -0.5*(np.linalg.norm(noisy_data_eight - my_soln_eight.T))**2/var # Gaussian
        log_like_eight = log_like_eight + l_like_eight

    # gaussian log prior

    a0 = -0.5*(np.linalg.norm((p[0]-np.array([mu_0,mu_1,mu_2,mu_2,mu_2,mu_2,mu_2,mu_2,mu_2]))/sigma_vec0))**2
    a1 = -0.5*(np.linalg.norm((p[1]-np.array([mu_0,mu_2,mu_1,mu_2,mu_2,mu_2,mu_2,mu_2,mu_2]))/sigma_vec1))**2
    a2 = -0.5*(np.linalg.norm((p[2]-np.array([mu_0,mu_2,mu_2,mu_1,mu_2,mu_2,mu_2,mu_2,mu_2]))/sigma_vec2))**2
    a3 = -0.5*(np.linalg.norm((p[3]-np.array([mu_0,mu_2,mu_2,mu_2,mu_1,mu_2,mu_2,mu_2,mu_2]))/sigma_vec3))**2
    a4 = -0.5*(np.linalg.norm((p[4]-np.array([mu_0,mu_2,mu_2,mu_2,mu_2,mu_1,mu_2,mu_2,mu_2]))/sigma_vec4))**2
    a5 = -0.5*(np.linalg.norm((p[5]-np.array([mu_0,mu_2,mu_2,mu_2,mu_2,mu_2,mu_1,mu_2,mu_2]))/sigma_vec5))**2
    a6 = -0.5*(np.linalg.norm((p[6]-np.array([mu_0,mu_2,mu_2,mu_2,mu_2,mu_2,mu_2,mu_1,mu_2]))/sigma_vec6))**2
    a7 = -0.5*(np.linalg.norm((p[7]-np.array([mu_0,mu_2,mu_2,mu_2,mu_2,mu_2,mu_2,mu_2,mu_1]))/sigma_vec7))**2

    log_prior = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7

    return -log_like_mono - log_like_pair - log_like_trio - log_like_seven - log_like_eight - log_prior

def support(q):
    p = np.zeros((m,m+1))
    p[0] = q[:9]
    p[1] = q[9:18]
    p[2] = q[18:27]
    p[3] = q[27:36]
    p[4] = q[36:45]
    p[5] = q[45:54]
    p[6] = q[54:63]
    p[7] = q[63:72]

    rt = True
    for i in range(m):
        rt &= 0.0<p[i][0]<2.0   #ri
        rt &= -3.0<p[i][i+1]<0.0  #aii

    return rt


def init():
    q = np.zeros(m*(m+1))
    q[:(m+1)] = np.array([mu_0,mu_1,mu_2,mu_2,mu_2,mu_2,mu_2,mu_2,mu_2]) + np.random.uniform(-0.01, 0.01)
    q[(m+1):2*(m+1)] = np.array([mu_0,mu_2,mu_1,mu_2,mu_2,mu_2,mu_2,mu_2,mu_2]) + np.random.uniform(-0.01, 0.01)
    q[2*(m+1):3*(m+1)] = np.array([mu_0,mu_2,mu_2,mu_1,mu_2,mu_2,mu_2,mu_2,mu_2]) + np.random.uniform(-0.01, 0.01)
    q[3*(m+1):4*(m+1)] = np.array([mu_0,mu_2,mu_2,mu_2,mu_1,mu_2,mu_2,mu_2,mu_2]) + np.random.uniform(-0.01, 0.01)
    q[4*(m+1):5*(m+1)] = np.array([mu_0,mu_2,mu_2,mu_2,mu_2,mu_1,mu_2,mu_2,mu_2]) + np.random.uniform(-0.01, 0.01)
    q[5*(m+1):6*(m+1)] = np.array([mu_0,mu_2,mu_2,mu_2,mu_2,mu_2,mu_1,mu_2,mu_2]) + np.random.uniform(-0.01, 0.01)
    q[6*(m+1):7*(m+1)] = np.array([mu_0,mu_2,mu_2,mu_2,mu_2,mu_2,mu_2,mu_1,mu_2]) + np.random.uniform(-0.01, 0.01)
    q[7*(m+1):8*(m+1)] = np.array([mu_0,mu_2,mu_2,mu_2,mu_2,mu_2,mu_2,mu_2,mu_1]) + np.random.uniform(-0.01, 0.01)
    return q

#### Remember to download pytwalk script before and put it
#### in the same directory

#### To verify if it works
burnin = 1
T=20

#### To get a convergent chain
#burnin = 10000
#T=50000


species = pytwalk.pytwalk(n=72,U=energy,Supp=support)
y0=init()
yp0=init()
species.Run(T,y0,yp0)

chain = species.Output
Energy = chain[:,-1]

### plot minus log posterior (all the chain)
pl.figure()
pl.plot(range(T+1),Energy)
pl.savefig('GLV/chain/energy.png')
pl.close()

### plot minus log posterior (without the burnin)
pl.figure()
pl.plot(range(T+1-burnin),Energy[burnin:])
pl.savefig('GLV/chain/energy_burn.png')
pl.close()


cadena=TemporaryFile()
np.save('GLV/chain/cadena',species.Output)

