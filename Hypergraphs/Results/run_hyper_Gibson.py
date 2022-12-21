# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 07:40:45 2019

@author: cimat
"""

import numpy as np


# pat = 'h'
# exec(open("gp_Parts_Gibson.py").read())
# exec(open("gp_read_samples_filter_Gibson.py").read())
# exec(open("gp_read_samples_filt_new_filter_Gibson.py").read())

# pat = 'uc'
# exec(open("gp_Parts_Gibson.py").read())
# exec(open("gp_read_samples_filter_Gibson.py").read())
# exec(open("gp_read_samples_filt_new_filter_Gibson.py").read())

#### Generate all models file using info from both patient types
# exec(open("gp_read_filt_new_all_samples_Gibson.py").read())

for pat in ['h','uc']:
    exec(open("gp_gen_inner_Pareto_purple.py").read())

