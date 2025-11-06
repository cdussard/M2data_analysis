# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 18:03:08 2024

@author: claire.dussard
"""
import pandas as pd 
import mne
import numpy as np
import matplotlib.pyplot as plt

cond = "pendule"

path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/Article_physio/analyses/reanalysisPerf/modbyFB/"
pval_perf = pd.read_csv(path+"df_"+cond+"_pval.csv").iloc[:, 1:]
pval_perf = pval_perf.to_numpy()

est_perf = pd.read_csv(path+"df_"+cond+"_estimate.csv").iloc[:, 1:]
est_perf = est_perf.to_numpy()

fmin = 3
fmax = 40
len_to_keep = fmax - fmin
df_subset = est_perf[:,0:len_to_keep+1]
df_pval_subset = pval_perf[:,0:len_to_keep+1]

cmap = "RdBu_r"


pvalue = 0.05
p_ag_corr = df_pval_subset*1096
p_perf_corr = df_pval_subset*1096
p_perf_corr=  mne.stats.fdr_correction(df_pval_subset)[1]


masked_perf = np.ma.masked_where((p_perf_corr > pvalue) , df_subset)


cmap = "RdBu_r"
my_cmap = discrete_cmap(13, cmap)
obj_channels=["Fp1","Fp2","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8",
"CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","O1","Oz","O2"]
info = mne.create_info(obj_channels,250,"eeg") #
info.set_montage(mne.channels.make_standard_montage('easycap-M1'))
freqs = np.arange(3,84,1)

vmax = 0.55
vmin = -vmax
fmin = 3
fmax = 7

cond = "pendule"
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/Article_physio/analyses/analyses_Nathalie/modelbyFB/"
est_perf_p = pd.read_csv(path+"df_pendule_estimate_norun.csv").iloc[:, 1:].to_numpy()

cond = "main"
est_perf_m = pd.read_csv(path+"df_"+cond+"_estimate_norun.csv").iloc[:, 1:].to_numpy()

cond = "mainvib"
est_perf_mv = pd.read_csv(path+"df_"+cond+"_estimate_norun.csv").iloc[:, 1:].to_numpy()

#cond = "mainvibavecvib"
#est_perf_mvib = pd.read_csv(path+"df_"+cond+"_estimate_norun.csv").iloc[:, 1:].to_numpy()

fmin = 3
fmax = 7
vmax = 0.6
vmin = -vmax
fig, axs = plt.subplots(1,3, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
plot_topomapV3(fmin,fmax,est_perf_p,vmin,vmax,my_cmap,axs,0)
plot_topomapV3(fmin,fmax,est_perf_m,vmin,vmax,my_cmap,axs,1)
plot_topomapV3(fmin,fmax,est_perf_mv,vmin,vmax,my_cmap,axs,2)
#plot_topomapV3(fmin,fmax,est_perf_mvib,vmin,vmax,my_cmap,axs,3)

vmax = 0.7
vmin = -vmax
fmin = 3
fmax = 4
fig, axs = plt.subplots(1,3, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
plot_topomapV3(fmin,fmax,est_perf_p,vmin,vmax,my_cmap,axs,0)
plot_topomapV3(fmin,fmax,est_perf_m,vmin,vmax,my_cmap,axs,1)
im = plot_topomapV3(fmin,fmax,est_perf_mv,vmin,vmax,my_cmap,axs,2)
fig.colorbar(im, location = 'right', shrink=0.5)

# plot_topomapV3(fmin,fmax,est_perf_mvib,vmin,vmax,my_cmap,axs,3)
