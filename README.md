# Assembly_arquetypes

This repository contains python and julia files to generate information and figures from the article Assembly archetypes in ecological communities. 

Posterior information directory contains Monte Carlo chains obtained from inferred data and Python scripts to generate marginal distributions plots. 

To run the Monte Carlo, you need to download the t-walk from:
<https://www.cimat.mx/~jac/twalk/>

To write assembly rules, we choose logic circuits written in the minimal Conjunctive Normal Form. So, given a coexistence hypergraph, we use python library pyeda to compute this form. The library documentation is available in:

<https://pyeda.readthedocs.io/en/latest/> 

version 0.28.0