# Assembly_arquetypes

__This repository contains__ python and julia files to generate information and figures from the article Assembly archetypes in ecological communities. 

__Posterior information directory__ contains Monte Carlo chains obtained from inferred data and Python scripts to generate marginal distributions plots. Also, the internal directory Friedman_data contains abundance observations for Friedman species community. MCMC_Friedman.py run the inference to obtain parameters from GLV's model. 

To run the Monte Carlo, you need to download the t-walk from:

<https://www.cimat.mx/~jac/twalk/>

__To write assembly rules__, we choose logic circuits written in the minimal Conjunctive Normal Form. So, given a coexistence hypergraph, we use python library pyeda to compute this form. The library documentation is available in:

<https://pyeda.readthedocs.io/en/latest/> 

version 0.28.0