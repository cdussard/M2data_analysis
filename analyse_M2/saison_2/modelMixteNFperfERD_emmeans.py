# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 23:30:39 2023

@author: claire.dussard
"""

import pandas as pd
import numpy as np
import imagesc

# Path to the directory containing the CSV files
#path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/fig_brian/correlERD_NFperf_colorgrid/"

#path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/fig_brian/correlERD_NFperf_colorgrid/FINAL_VERSIONS_all3FB/emmeans/perf_etalonne/data/"#que juste 35hz
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/fig_brian/correlERD_NFperf_colorgrid/FINAL_VERSIONS_all3FB/actualData/perf_etalonne_4/"
#df_global = pd.read_csv(path + "df_global_estimate_0.csv", header=None, delimiter=",").iloc[1:, 1:].values
# Read the CSV files
for lim in ["0","25","50","75","100"]:
    df_global = pd.read_csv(path + "df_"+lim+"perf_estimate_.csv", header=None, delimiter=",").iloc[1:, 1:].values
    df_global = df_global.astype(float)
    print(df_global.max())
    plot_topomapV2(3,7,df_global,vmin,vmax,my_cmap)
    plot_topomapV2(8,30,df_global,vmin,vmax,my_cmap)
    plt.show()


# Convert to numeric values and float dtype
#imagesc.plot(df_global,cmap="RdBu_r")

vmax = 0.3
vmin = -0.3

#A BIEN CHOISIR
vmax = 0.3
vmin = -vmax
my_cmap = discrete_cmap(13, cmap)
plot_topomapV2(3,7,df_global,vmin,vmax,my_cmap)
#plot_topomapV2(8,12,df_global,vmin,vmax,my_cmap)
plot_topomapV2(8,30,df_global,vmin,vmax,my_cmap)
plt.show()
#←raw_signal.plot(block=True)

# elecs = elec_leg 
# #plt.subplots_adjust(wspace=0.2, hspace=0.05)
# fig, axs = plt.subplots(1,1, sharey=True,sharex=True, figsize=(20, 7),constrained_layout=True)
# freq_leg = np.arange(3,84,4)
# freq_leg_str =[str(f) for f in freq_leg]
# plt.xticks(np.linspace(0,1,21),freq_leg_str)
# x8Hz = 0.061
# x30Hz = 0.34
# col = "black"
# ls = "--"
# lw = 0.7
# axs.axvline(x=x8Hz,color=col,ls=ls,lw=lw)
# axs.axvline(x=x30Hz,color=col,ls=ls,lw=lw)
# plt.yticks(np.linspace(1/(len(elecs)*2.5),1-1/(len(elecs)*2.5),len(elecs)),elecs.iloc[::-1])
# for elecPos in [0.107,0.286,0.428,0.608,0.75,0.9293]:
#     axs.axhline(y=elecPos,color="dimgray",lw=0.25)
# img = axs.imshow(df_global, extent=[0, 1, 0, 1],cmap=cmap, aspect='auto',interpolation='none',vmin=vmin,vmax=vmax,label="agency") 
# fig.colorbar(img, location = 'right')
# plt.show()


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


