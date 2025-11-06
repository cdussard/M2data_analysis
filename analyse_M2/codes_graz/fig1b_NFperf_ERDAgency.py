# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 18:14:06 2024

@author: claire.dussard
"""

import pandas as pd
import mne
import numpy as np
import imagesc
from functions_graz.plot_functions import *
import matplotlib.pyplot as plt

# Path to the directory containing the CSV files
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/Article_physio/analyses/Agency/"

obj_channels=["Fp1","Fp2","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8",
"CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","O1","Oz","O2"]
reorganize_data_elec = ["Fp1","F7","F3","FC5","FC1","T7","C3","CP5","CP1","P7","P3","O1"
                        "Fz","Cz","Pz","Oz"
                        "Fp2","F4", "F8","FC2","FC6","T8","CP2","CP6","P4","P8","O2"
                       
                ]
# Read the CSV files

estimate_ERD = pd.read_csv(path + "estimate_modagencyERD_ERD.csv", header=None, delimiter=",").iloc[1:, 1:].values
pval_ERD = pd.read_csv(path + "pval_modagencyERD_ERD.csv", header=None, delimiter=",").iloc[1:, 1:].values

# estimate_ERD = pd.read_csv(path + "estimate_modagencyERD2_ERD.csv", header=None, delimiter=",").iloc[1:, 1:].values
# pval_ERD = pd.read_csv(path + "pval_modagencyERD2_ERD.csv", header=None, delimiter=",").iloc[1:, 1:].values

estimate_ERD = pd.read_csv(path + "estimate_modFBERD_ERD.csv", header=None, delimiter=",").iloc[1:, 1:].values
pval_ERD = pd.read_csv(path + "pval_modFBERD_ERD.csv", header=None, delimiter=",").iloc[1:, 1:].values


estimate_ERD = pd.read_csv(path + "estimate_modFBERD_ERD2.csv", header=None, delimiter=",").iloc[1:, 1:].values
pval_ERD = pd.read_csv(path + "pval_modFBERD_ERD2.csv", header=None, delimiter=",").iloc[1:, 1:].values

pval_ERD = pval_ERD.astype(float)
estimate_ERD = estimate_ERD.astype(float)

# new_order = [obj_channels.index(elec) for elec in reorganize_data_elec]
# reorganized_estimate = estimate_ERD[new_order, :]
# reorganized_pval = pval_ERD[new_order, :]

#subset from fmin to fmax
fmin = 3
fmax = 40
len_to_keep = fmax - fmin
pval_ERD = pval_ERD[:,0:len_to_keep+1]
estimate_ERD = estimate_ERD[:,0:len_to_keep+1]
print(pval_ERD.min())



#FDR correction
corrected_pval_erd= mne.stats.fdr_correction(pval_ERD)[1]
print(corrected_pval_erd.min())


masked_pval_ERD = np.ma.masked_where((corrected_pval_erd > 0.05) , estimate_ERD)

masked_pval_ERD = np.ma.masked_where((pval_ERD > 0.01) , estimate_ERD)


vmin = -0.2
vmax = -vmin
cmap = "RdBu_r"

# Create a subplot with 2 rows and 3 columns
fig, axs = plt.subplots(1, 4, figsize=(16, 8))
plot_topomapV3(3,7,estimate_ERD,vmin,vmax,cmap,axs,0)
plot_topomapV3(8,12,estimate_ERD,vmin,vmax,cmap,axs,1)
plot_topomapV3(13,20,estimate_ERD,vmin,vmax,cmap,axs,2)
im = plot_topomapV3(21,30,estimate_ERD,vmin,vmax,cmap,axs,3)
fig.colorbar(im, location = 'right',ax=axs,shrink=0.5)
plot_topomapV2(8,30,estimate_ERD,vmin,vmax,cmap)

vmin = -0.4
vmax = -vmin
plot_fig_elec(masked_pval_ERD,vmin,vmax,cmap,None,elecs)

plot_fig_elec(estimate_ERD,vmin,vmax,cmap,pval_ERD,elecs)


cmap = "Reds"
vmin = 0
vmax = 0.2
plot_fig_elec(pval_ERD,vmin,vmax,cmap,pval_ERD,obj_channels)


# reorganized_estimate = estimate_ERD[new_order, :]
# reorganized_pval = pval_ERD[new_order, :]

# plot_fig_elec(reorganized_estimate,vmin,vmax,cmap,reorganized_pval,elecs)