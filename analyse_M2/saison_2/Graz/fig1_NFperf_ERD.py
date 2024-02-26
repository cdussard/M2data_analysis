# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:57:08 2024

@author: claire.dussard
"""
import pandas as pd
import mne
import numpy as np
import imagesc
from functions_graz import *

# Path to the directory containing the CSV files
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/Article_physio/analyses/reanalysisPerf/"


# Read the CSV files
estimate_ERD = pd.read_csv(path + "estimate_predPerfbyERD_run_vib.csv", header=None, delimiter=",").iloc[1:, 1:].values
pval_ERD = pd.read_csv(path + "pval_predPerfbyERD_run_vib.csv", header=None, delimiter=",").iloc[1:, 1:].values
pval_ERD = pval_ERD.astype(float)
estimate_ERD = estimate_ERD.astype(float)
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



vmin = -0.45
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

vmin = -0.6
vmax = -vmin
plot_fig_elec(masked_pval_ERD,vmin,vmax,cmap)

#===================== CONTROLE = figure sans vibrations =========================
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/Article_physio/analyses/reanalysisPerf/"
estimate_ERD = pd.read_csv(path + "estimate_predPerfbyERD_run_novib.csv", header=None, delimiter=",").iloc[1:, 1:].values
pval_ERD = pd.read_csv(path + "pval_predPerfbyERD_run_novib.csv", header=None, delimiter=",").iloc[1:, 1:].values
pval_ERD = pval_ERD.astype(float)
estimate_ERD = estimate_ERD.astype(float)
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



vmin = -0.45
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

vmin = -0.6
vmax = -vmin
plot_fig_elec(masked_pval_ERD,vmin,vmax,cmap)