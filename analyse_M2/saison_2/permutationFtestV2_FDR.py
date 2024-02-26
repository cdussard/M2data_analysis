# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 20:02:10 2023

@author: claire.dussard
"""

import pandas as pd
import numpy as np
import mne
import imagesc

path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/rdom_scriptsData/allElecFreq_VSZero/versionJuin2023_elecFixed/"

p_pend = pd.read_csv(path+"p_pend.csv").iloc[:, 1:]
p_main = pd.read_csv(path+"p_main.csv").iloc[:, 1:]
p_mIll = pd.read_csv(path+"p_mainIllusion.csv").iloc[:, 1:]

p_pend = p_pend.to_numpy()
p_main = p_main.to_numpy()
p_mIll = p_mIll.to_numpy()

pend = pd.read_csv(path+"dcohen_mainIllusion.csv").iloc[:, 1:]
main = pd.read_csv(path+"dcohen_main.csv").iloc[:, 1:]
mIll = pd.read_csv(path+"dcohen_pend.csv").iloc[:, 1:]

pend = pend.to_numpy()
main = main.to_numpy()
mIll = mIll.to_numpy()



pvalue = 0.05
p_pend_corr=  mne.stats.fdr_correction(p_pend)[1]
p_main_corr=  mne.stats.fdr_correction(p_main)[1]
p_mIll_corr=  mne.stats.fdr_correction(p_mIll)[1]

masked_p = np.ma.masked_where((p_pend_corr > pvalue) , pend)
masked_m = np.ma.masked_where((p_main_corr > pvalue) , main)
masked_mi = np.ma.masked_where((p_mIll_corr > pvalue) , mIll)

# imagesc.plot(-masked_p,cmap="Blues")
# imagesc.plot(-masked_m,cmap="Blues")
# imagesc.plot(-masked_mi,cmap="Blues")

import matplotlib.pyplot as plt
elec_leg = pd.read_csv(path+"dcohen_mainIllusion.csv").iloc[:, 0]
gridspec_kw={'width_ratios': [1,1,1],
                           'height_ratios': [1],
                       'wspace': 0.05,#constrained_layout=True
                       'hspace': 0.05}
fig, axs = plt.subplots(1,3, sharey=True,sharex=True, figsize=(20, 7),gridspec_kw=gridspec_kw,constrained_layout=True)
vmin = 0#0.56
vmax = 2.13#2
img = axs[0].imshow(-masked_p, extent=[0, 1, 0, 1],cmap="Blues", aspect='auto',interpolation='none',vmin=vmin,vmax=vmax,label="pendulum")
axs[0].text(0.12, 1.02, 'Virtual pendulum')

axs[1].imshow(-masked_m, extent=[0, 1, 0, 1],cmap="Blues", aspect='auto',interpolation='none',vmin=vmin,vmax=vmax)
axs[1].text(0.12, 1.02, 'Virtual hand')
axs[2].imshow(-masked_mi, extent=[0, 1, 0, 1],cmap="Blues", aspect='auto',interpolation='none',vmin=vmin,vmax=vmax)
axs[2].text(0.12, 1.02, 'Virtual hand with vibrations')
fig.colorbar(img, location = 'right')
elecs = elec_leg 
#plt.subplots_adjust(wspace=0.2, hspace=0.05)
freq_leg = np.arange(3,84,4)
freq_leg_str =[str(f) for f in freq_leg]
plt.xticks(np.linspace(0,1,21),freq_leg_str)
x8Hz = 0.061
x30Hz = 0.3415
col = "black"
ls = "--"
lw = 0.7
for ax in axs.flat:
    ax.axvline(x=x8Hz,color=col,ls=ls,lw=lw)
    ax.axvline(x=x30Hz,color=col,ls=ls,lw=lw)
plt.yticks(np.linspace(1/(len(elecs)*2.5),1-1/(len(elecs)*2.5),len(elecs)),elecs.iloc[::-1])
for ax in axs.flat:
    for elecPos in [0.107,0.286,0.428,0.608,0.75,0.9293]:
        ax.axhline(y=elecPos,color="dimgray",lw=0.25)
#plt.tight_layout(pad=0.04) 
plt.show()


#raw
import matplotlib.pyplot as plt
elec_leg = pd.read_csv(path+"dcohen_mainIllusion.csv").iloc[:, 0]
gridspec_kw={'width_ratios': [1,1,1],
                           'height_ratios': [1],
                       'wspace': 0.05,#constrained_layout=True
                       'hspace': 0.05}
fig, axs = plt.subplots(1,3, sharey=True,sharex=True, figsize=(20, 7),gridspec_kw=gridspec_kw,constrained_layout=True)
vmin = 0.44
vmax = 2.13
img = axs[0].imshow(-pend, extent=[0, 1, 0, 1],cmap="Blues", aspect='auto',interpolation='none',vmin=vmin,vmax=vmax,label="pendulum")
axs[0].text(0.12, 1.02, 'Virtual pendulum')

axs[1].imshow(-main, extent=[0, 1, 0, 1],cmap="Blues", aspect='auto',interpolation='none',vmin=vmin,vmax=vmax)
axs[1].text(0.12, 1.02, 'Virtual hand')
axs[2].imshow(-mIll, extent=[0, 1, 0, 1],cmap="Blues", aspect='auto',interpolation='none',vmin=vmin,vmax=vmax)
axs[2].text(0.12, 1.02, 'Virtual hand with vibrations')
fig.colorbar(img, location = 'right')
elecs = elec_leg 
#plt.subplots_adjust(wspace=0.2, hspace=0.05)
freq_leg = np.arange(3,84,4)
freq_leg_str =[str(f) for f in freq_leg]
plt.xticks(np.linspace(0,1,21),freq_leg_str)
x8Hz = 0.061
x30Hz = 0.3415
col = "black"
ls = "--"
lw = 0.7
for ax in axs.flat:
    ax.axvline(x=x8Hz,color=col,ls=ls,lw=lw)
    ax.axvline(x=x30Hz,color=col,ls=ls,lw=lw)
plt.yticks(np.linspace(1/(len(elecs)*2.5),1-1/(len(elecs)*2.5),len(elecs)),elecs.iloc[::-1])
for ax in axs.flat:
    for elecPos in [0.107,0.286,0.428,0.608,0.75,0.9293]:
        ax.axhline(y=elecPos,color="dimgray",lw=0.25)
#plt.tight_layout(pad=0.04) 
plt.show()