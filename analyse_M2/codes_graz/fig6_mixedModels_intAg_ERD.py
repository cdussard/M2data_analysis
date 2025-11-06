# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 20:12:05 2024

@author: claire.dussard
"""

import pandas as pd
import mne
import numpy as np
import imagesc
from functions_graz.plot_functions import *

# Path to the directory containing the CSV files
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/Article_physio/analyses/Agency/"
#ALL IS NO VIB

# Read the CSV files

model = "m6"

pval_intFB = pd.read_csv(path + "pval_"+model+"_intERDFB.csv", header=None, delimiter=",").iloc[1:, 1:].values

estimate_intAgency = pd.read_csv(path + "estimate_"+model+"_intERDAgency.csv", header=None, delimiter=",").iloc[1:, 1:].values
pval_intAgency = pd.read_csv(path + "pval_"+model+"_intERDAgency.csv", header=None, delimiter=",").iloc[1:, 1:].values#POURQUOI ILS SONT IDENTIQUES

pval_intFB = pval_intFB.astype(float)
estimate_intAgency = estimate_intAgency.astype(float)
pval_intAgency = pval_intAgency.astype(float)

#subset from fmin to fmax
fmin = 3
fmax = 40
len_to_keep = fmax - fmin
est_ag = estimate_intAgency[:,0:len_to_keep+1]
pval_intfb = pval_intFB[:,0:len_to_keep+1]
pval_intag = pval_intAgency[:,0:len_to_keep+1]



vmin = 0#-0.35
vmax = 7
cmap = "Greens"


#FDR correction

corrected_pval_intfb = mne.stats.fdr_correction(pval_intfb)[1]
corrected_pval_intag = mne.stats.fdr_correction(pval_intag)[1]
print(corrected_pval_intfb.min())
print(corrected_pval_intag.min())

# # pvalue = 0.05
# log_pval_ERD = -np.log10(corrected_pval_erd)
# log_pval_intFB = -np.log10(corrected_pval_int)

# pvalue = 0.05

log_pval_intFB = -np.log10(pval_intfb)
log_pval_intag = -np.log10(pval_intag)

# log_pval_intFB = -np.log10(corrected_pval_intfb)
# log_pval_intag = -np.log10(corrected_pval_intag)


masked_log_pval_intag = np.ma.masked_where((log_pval_intag < 1.29) , log_pval_intag)
masked_log_pval_intFB = np.ma.masked_where((log_pval_intFB < -np.log10(0.05)) , log_pval_intFB)


masked_est_intag = np.ma.masked_where((log_pval_intag < 1.29) , est_ag)
plot_fig_elec(masked_log_pval_ERD,vmin,vmax,cmap)
#plot int FB
vmin = 0#-0.35
vmax =4
cmap = "Greens"
plot_fig_elec(masked_log_pval_intag,vmin,vmax,cmap,None,elecs)
plot_fig_elec(masked_log_pval_intFB,vmin,vmax,cmap,None,elecs)

vmin = -0.2
vmax =0.2
cmap = "RdBu_r"
plot_fig_elec(masked_est_intag,vmin,vmax,cmap,None,elecs)

#topomap


freqs = np.arange(3,84,1)
obj_channels=["Fp1","Fp2","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8",
"CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","O1","Oz","O2"]
info = mne.create_info(obj_channels,250,"eeg")
info.set_montage(mne.channels.make_standard_montage('easycap-M1'))

cmap = "Greens"
vmin = 0
vmax = 2
plot_topomapV2(3,7,log_pval_intag,vmin,vmax,cmap)
plot_topomapV2(3,7,log_pval_intFB,vmin,vmax,cmap)


cmap = "RdBu_r"
vmin = -0.1
vmax = 0.1
plot_topomapV2(3,7,est_ag,vmin,vmax,cmap)