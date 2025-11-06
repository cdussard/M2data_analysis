# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:57:26 2024

@author: claire.dussard
"""

import pandas as pd
import mne
import numpy as np
import imagesc
from functions_graz import *

# Path to the directory containing the CSV files
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/Article_physio/analyses/analyses_Nathalie/"


# Read the CSV files
pval_ERD = pd.read_csv(path + "pval_ERDprefPerf_datasetsansvib.csv", header=None, delimiter=",").iloc[1:, 1:].values
Fval_ERD = pd.read_csv(path + "Fval_ERDprefPerf_datasetsansvib.csv", header=None, delimiter=",").iloc[1:, 1:].values

pval_intFB = pd.read_csv(path + "pval_IntFBERDprefPerf_datasetsansvib.csv", header=None, delimiter=",").iloc[1:, 1:].values
pval_intFBvib = pd.read_csv(path + "pval_IntFBERDprefPerf_datasetavecvib.csv", header=None, delimiter=",").iloc[1:, 1:].values

Fval_intFB = pd.read_csv(path + "Fval_IntFBERDprefPerf_datasetsansvib.csv", header=None, delimiter=",").iloc[1:, 1:].values
Fval_intFBvib = pd.read_csv(path + "Fval_IntFBERDprefPerf_datasetavecvib.csv", header=None, delimiter=",").iloc[1:, 1:].values#POURQUOI ILS SONT IDENTIQUES

pval_ERD = pval_ERD.astype(float)
Fval_ERD = Fval_ERD.astype(float)
pval_intFB = pval_intFB.astype(float)
pval_intFBvib = pval_intFBvib.astype(float)
Fval_intFB = Fval_intFB.astype(float)
Fval_intFBvib = Fval_intFBvib.astype(float)

#subset from fmin to fmax
fmin = 3
fmax = 40
len_to_keep = fmax - fmin
pval_erd = pval_ERD[:,0:len_to_keep+1]
pval_int = pval_intFB[:,0:len_to_keep+1]
pval_intvib = pval_intFBvib[:,0:len_to_keep+1]

Fval_int = Fval_intFB[:,0:len_to_keep+1]
Fval_intvib = Fval_intFBvib[:,0:len_to_keep+1]
print(pval_int.min())



vmin = 0#-0.35
vmax = 7
cmap = "Greens"


#FDR correction

corrected_pval_int = mne.stats.fdr_correction(pval_int)[1]
corrected_pval_erd = mne.stats.fdr_correction(pval_erd)[1]
print(corrected_pval_int.min())

# # pvalue = 0.05
# log_pval_ERD = -np.log10(corrected_pval_erd)
# log_pval_intFB = -np.log10(corrected_pval_int)

# pvalue = 0.05
log_pval_ERD = -np.log10(pval_erd)
log_pval_intFB = -np.log10(pval_int)
log_pval_intFBvib = -np.log10(pval_intvib)

masked_log_pval_ERD = np.ma.masked_where((log_pval_ERD < 1.29) , log_pval_ERD)
masked_log_pval_intFB = np.ma.masked_where((log_pval_intFB < -np.log10(0.05)) , log_pval_intFB)
masked_log_pval_intFBvib = np.ma.masked_where((log_pval_intFBvib < 1.29) , log_pval_intFBvib)


#plot_fig_elec(masked_log_pval_ERD,vmin,vmax,cmap)
#plot int FB
vmin = 0#-0.35
vmax =4

plot_fig_elec(masked_log_pval_ERD,vmin,vmax,cmap,None,elecs)
plot_fig_elec(masked_log_pval_intFB,vmin,vmax,cmap,None,elecs)
plot_fig_elec(masked_log_pval_intFBvib,vmin,vmax,cmap,None,elecs)

plot_fig_elec(log_pval_intFB,vmin,vmax,cmap)
plot_fig_elec(Fval_int,vmin,vmax,cmap)



#topomap


freqs = np.arange(3,84,1)
obj_channels=["Fp1","Fp2","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8",
"CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","O1","Oz","O2"]
info = mne.create_info(obj_channels,250,"eeg")
info.set_montage(mne.channels.make_standard_montage('easycap-M1'))

vmin = 0
vmax = 3
plot_topomapV2(3,3,log_pval_intFB,vmin,vmax,cmap)
plot_topomapV2(4,4,log_pval_intFB,vmin,vmax,cmap)
plot_topomapV2(3,4,log_pval_intFB,vmin,vmax,cmap)

vmin = 0
vmax = 13
plot_topomapV2(3,4,Fval_ERD,vmin,vmax,cmap)
vmin = 0
vmax = 4
plot_topomapV2(3,4,Fval_intFB,vmin,vmax,cmap)



vmin = 0
vmax = 7
fig, axs = plt.subplots(1,2, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
plot_topomapV3(3,4,Fval_ERD,vmin,vmax,cmap,axs,0)
im = plot_topomapV3(3,4,Fval_intvib,vmin,vmax,cmap,axs,1)
fig.colorbar(im, location = 'right', shrink=0.7)



fig, axs = plt.subplots(1,2, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
plot_topomapV3(3,4,Fval_intFB,vmin,vmax,cmap,axs,0)
im = plot_topomapV3(3,4,Fval_intvib,vmin,vmax,cmap,axs,1)
fig.colorbar(im, location = 'right', shrink=0.7)


# vmin = 0
# vmax = 2
# fig, axs = plt.subplots(1,2, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
# plot_topomapV3(3,4,log_pval_intFB,vmin,vmax,cmap,axs,0)

# plot_topomapV3(8,30,log_pval_intFB,vmin,vmax,cmap,axs,1)
# my_cmap = discrete_cmap(13, 'Reds')




# vmax = 0.4
# vmin = -vmax
# fig, axs = plt.subplots(1,5, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
# plot_topomapV3(3,7,df_subset,vmin,vmax,cmap,axs[0],0)
# plot_topomapV3(8,12,df_subset,vmin,vmax,cmap,axs,1)
# plot_topomapV3(12,15,df_subset,vmin,vmax,cmap,axs,2)
# plot_topomapV3(13,20,df_subset,vmin,vmax,cmap,axs,3)
# im = plot_topomapV3(21,30,df_subset,vmin,vmax,cmap,axs,4)
# fig.colorbar(im, location = 'right', shrink=0.5)

# plot_topomapV2(8,30,df_subset,-0.35,0.35,cmap)






# #si on affiche les F values
# Fval_intFB
# vmin = 0
# vmax = 5
# fig, axs = plt.subplots(1,2, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
# plot_topomapV3(3,4,Fval_intFB,vmin,vmax,cmap,axs,0)
# im = plot_topomapV3(3,7,Fval_intFB,vmin,vmax,cmap,axs,1)
# fig.colorbar(im, location = 'right', shrink=0.7)
