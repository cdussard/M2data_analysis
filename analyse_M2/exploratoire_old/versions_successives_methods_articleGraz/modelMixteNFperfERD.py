# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 13:35:40 2023

@author: claire.dussard
"""

import pandas as pd
import mne
import numpy as np


# Path to the directory containing the CSV files
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/fig_brian/correlERD_NFperf_colorgrid/"


# Read the CSV files
df_global = pd.read_csv(path + "df_global_estimateFBseul_v3_2runs_NFperf.csv", header=None, delimiter=",").iloc[1:, 1:].values
df_global_pval = pd.read_csv(path + "df_global_pvalFBseul_v3_2runs_NFperf.csv", header=None, delimiter=",").iloc[1:, 1:].values

df_global = df_global.astype(float)
df_global_pval = df_global_pval.astype(float)

print(df_global.max())

# Convert to numeric values and float dtype

pvalue = 0.001
masked_global = np.ma.masked_where((df_global_pval > pvalue) , df_global)
masked_global.data
#♣df_global[df_global_pval < pvalue]

vmin = -0.32
vmax = -vmin
cmap = "RdBu_r"


import matplotlib.pyplot as plt
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/rdom_scriptsData/allElecFreq_VSZero/versionJuin2023_elecFixed/"
elec_leg = pd.read_csv(path+"dcohen_mainIllusion.csv").iloc[:, 0]

elecs = elec_leg 
#plt.subplots_adjust(wspace=0.2, hspace=0.05)
fig, axs = plt.subplots(1,1, sharey=True,sharex=True, figsize=(20, 7),constrained_layout=True)
freq_leg = np.arange(3,84,4)
freq_leg_str =[str(f) for f in freq_leg]
plt.xticks(np.linspace(0,1,21),freq_leg_str)
x8Hz = 0.061
x30Hz = 0.34
col = "black"
ls = "--"
lw = 0.7
axs.axvline(x=x8Hz,color=col,ls=ls,lw=lw)
axs.axvline(x=x30Hz,color=col,ls=ls,lw=lw)
plt.yticks(np.linspace(1/(len(elecs)*2.5),1-1/(len(elecs)*2.5),len(elecs)),elecs.iloc[::-1])
for elecPos in [0.107,0.286,0.428,0.608,0.75,0.9293]:
    axs.axhline(y=elecPos,color="dimgray",lw=0.25)
img = axs.imshow(masked_global, extent=[0, 1, 0, 1],cmap=cmap, aspect='auto',interpolation='none',vmin=vmin,vmax=vmax,label="agency") 
fig.colorbar(img, location = 'right')
plt.show()



obj_channels=["Fp1","Fp2","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8",
"CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","O1","Oz","O2"]

info = mne.create_info(obj_channels,250,"eeg") #
info.set_montage(mne.channels.make_standard_montage('easycap-M1'))
freqs = np.arange(3,84,1)


def plot_topomapV2(fmin,fmax,masked_global,vmin,vmax,cmap):
    print(freqs[fmin-3:fmax-2])
    masked_global_freq = masked_global[:,fmin-3:fmax-2]
    mean = np.mean(masked_global_freq,axis=1)
    print(mean.max())
    print(mean.min())
    fig,ax = plt.subplots(ncols=1)
    im,cm   = mne.viz.plot_topomap(mean,info, axes=ax,show=False,vmin=vmin,vmax=vmax,cmap=cmap)#,border=0,sphere='eeglab')   
    fig.colorbar(im, location = 'right')
    return fig,mean


#A BIEN CHOISIR
vmax = 0.22#0.3
vmin = -vmax
my_cmap = discrete_cmap(13, cmap)
plot_topomapV2(3,7,df_global,vmin,vmax,my_cmap)
plot_topomapV2(8,12,df_global,vmin,vmax,my_cmap)
plot_topomapV2(12,15,df_global,vmin,vmax,my_cmap)
plot_topomapV2(13,20,df_global,vmin,vmax,my_cmap)
#plot_topomapV2(20,30,df_global,vmin,vmax,my_cmap)
#plot_topomapV2(13,30,df_global,vmin,vmax,my_cmap)
plot_topomapV2(8,30,df_global,vmin,vmax,my_cmap)
raw_signal.plot(block=True)

 