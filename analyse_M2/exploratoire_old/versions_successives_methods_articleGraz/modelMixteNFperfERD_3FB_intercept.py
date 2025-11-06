# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 18:20:09 2023

@author: claire.dussard
"""


import pandas as pd
import numpy as np
import imagesc

# Path to the directory containing the CSV files
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/fig_brian/correlERD_NFperf_colorgrid/v2_brian_eachFB/"

cond = "mainvib"

# write.csv(df_main_estimate_pos,"df_main_estimate_emmeans_pos.csv")
# write.csv(df_mainvib_estimate_pos,"df_mainvib_estimate_emmeans_pos.csv")
# write.csv(df_pendule_estimate_pos,"df_pendule_estimate_emmeans_pos.csv")

# write.csv(df_pendule_estimate_neg,"df_main_estimate_emmeans_neg.csv")
# write.csv(df_mainvib_estimate_neg,"df_mainvib_estimate_emmeans_neg.csv")
# write.csv(df_pendule_estimate_neg,"df_pendule_estimate_emmeans_neg.csv")
# Read the CSV files
df_global = pd.read_csv(path + "df_"+cond+"_estimate_intercept.csv", header=None, delimiter=",").iloc[1:, 1:].values
df_global_pval = pd.read_csv(path + "df_"+cond+"_pval_intercept.csv", header=None, delimiter=",").iloc[1:, 1:].values


df_global = df_global.astype(float)
df_global_pval = df_global_pval.astype(float)

# Convert to numeric values and float dtype

pvalue = 0.0001
masked_global = np.ma.masked_where((df_global_pval > pvalue) , df_global)
masked_global.data
df_global[df_global_pval < pvalue]

#imagesc.plot(masked_global,cmap="RdBu_r")
# imagesc.plot(df_global,cmap="RdBu_r")
#raw_signal.plot(block=True)

import matplotlib.pyplot as plt
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/rdom_scriptsData/allElecFreq_VSZero/versionJuin2023_elecFixed/"
elec_leg = pd.read_csv(path+"dcohen_mainIllusion.csv").iloc[:, 0]

vmax=0.4#0.04
vmin=-vmax

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
axs.imshow(df_global, extent=[0, 1, 0, 1],cmap="RdBu_r", aspect='auto',interpolation='none',vmin=vmin,vmax=vmax,label="agency") 
img = axs.imshow(df_global, extent=[0, 1, 0, 1], cmap="RdBu_r", aspect='auto', interpolation='none', vmin=vmin, vmax=vmax, label="agency")
fig.colorbar(img, location = 'right')
plt.show()
raw_signal.plot(block=True)

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
    ax_x_start = 0.95
    ax_x_width = 0.04
    ax_y_start = 0.1
    ax_y_height = 0.9
    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    
    clb = fig.colorbar(im, cax=cbar_ax)
    clb.ax.set_title("unit_label",fontsize=12)
    return fig,mean

my_cmap = discrete_cmap(13, 'RdBu_r')#colormap tres custom : blanc pour val negatives et val positives faibles 

vmax = 0.3
vmin = -vmax # Adjust the min and max values as needed

plot_topomapV2(3,7,df_global,vmin,vmax,my_cmap)
plot_topomapV2(8,12,df_global,vmin,vmax,my_cmap)
plot_topomapV2(8,30,df_global,vmin,vmax,my_cmap)
raw_signal.plot(block=True)
