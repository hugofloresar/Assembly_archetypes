# Assembly_arquetypes

__This repository contains__ python and julia files to generate information and figures from the article Assembly archetypes in ecological communities. 

__Posterior information directory__ contains Monte Carlo chains obtained from inferred data and Python scripts to generate marginal distributions plots. Also, the internal directory Friedman_data contains abundance observations for Friedman species community. MCMC_Friedman.py run the inference to obtain parameters from GLV's model. 

To run the Monte Carlo, you need to download the t-walk from:

<https://www.cimat.mx/~jac/twalk/>

The internal directory Droso_results contains MCMC results from fitting a GLV model to Drosophila abundance data. Supplementary Figure 8 may be reproduced from Python script Suppl_Fig8.py 

The internal directory Gibson_results contains MCMC results from Gibson data. We have used the google_colab notebooks provided by the authors in 

<https://github.com/gerberlab/MDSINE2_Paper/tree/master/google_colab>

The internal directory Gibson_plots contains the information to generate Supplementary Figures 9 and 10. Use the script Gibson_plots.py

___________

__To write assembly rules__, we choose logic circuits written in the minimal Conjunctive Normal Form. So, given a coexistence hypergraph, we use python library pyeda to compute this form. The library documentation is available in:

<https://pyeda.readthedocs.io/en/latest/> 

version 0.28.0