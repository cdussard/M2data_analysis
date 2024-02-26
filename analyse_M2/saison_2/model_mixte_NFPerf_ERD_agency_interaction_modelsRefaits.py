# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 17:57:41 2024

@author: claire.dussard
"""

import pandas as pd
import mne
import numpy as np
import imagesc
import matplotlib.pyplot as plt
import scipy

path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/rdom_scriptsData/allElecFreq_VSZero/versionJuin2023_elecFixed/"
elec_leg = pd.read_csv(path+"dcohen_mainIllusion.csv").iloc[:, 0]

freqs = np.arange(3,84,1)
obj_channels=["Fp1","Fp2","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8",
"CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","O1","Oz","O2"]
info = mne.create_info(obj_channels,250,"eeg") #
info.set_montage(mne.channels.make_standard_montage('easycap-M1'))
freqs = np.arange(3,84,1)

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
def plot_topomapV3(fmin,fmax,masked_global,vmin,vmax,cmap,axs,i):
    print(freqs[fmin-3:fmax-2])
    masked_global_freq = masked_global[:,fmin-3:fmax-2]
    mean = np.mean(masked_global_freq,axis=1)
    print(mean.max())
    print(mean.min())
    im,cm   = mne.viz.plot_topomap(mean,info, axes=axs[i],show=False,vmin=vmin,vmax=vmax,cmap=cmap)#,border=0,sphere='eeglab')   
    return im

cmap = "RdBu_r"
my_cmap = discrete_cmap(13, cmap)


path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/fig_brian/nouvelleFigureH4_Agency_ERD_perf/df_globals_REML_T_anovalmerTest/"
elec_leg = pd.read_csv(path+"df_global_pval_v3_2runs_NFperfAgTogether_agperf.csv").iloc[:, 0]


pval_perfag = pd.read_csv(path+"df_global_pval_v3_2runs_NFperfAgTogether_agperf.csv").iloc[:, 1:]
pval_perfag = pval_perfag.to_numpy()


est_perfag = pd.read_csv(path+"df_global_estimate_v3_NFperfAgTogether_agperf.csv").iloc[:, 1:]
est_perfag = est_perfag.to_numpy()

fmin = 3
fmax = 40
len_to_keep = fmax - fmin
df_subset_perfag = est_perfag[:,0:len_to_keep+1]
df_pval_subset_perfag = pval_perfag[:,0:len_to_keep+1]


cmap = "RdBu_r"

pvalue = 0.05
p_perfag_corr=  mne.stats.fdr_correction(df_pval_subset_perfag)[1]

masked_perfag = np.ma.masked_where((df_pval_subset_perfag > pvalue) , df_subset_perfag)

cmap = "RdBu_r"
vmin =-0.09 #-0.35
vmax =0.09 #0.35#0.01
elecs = elec_leg 
fig, axs = plt.subplots(1,1, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
freq_leg = np.arange(3,40,4)
freq_leg_str =[str(f) for f in freq_leg]
pos_freq = np.linspace(0.015,0.985,len(freq_leg))
for i in range(len(pos_freq)):
    print(i)
    if i<3:
        pos_freq[i] = pos_freq[i]*(1-i*0.014)
    elif i==3:
        pos_freq[i] = pos_freq[i]*(1-i*0.012)
    elif i ==4:
        pos_freq[i] = pos_freq[i]*(1-i*0.008)
    elif i==len(pos_freq)-1:
        print("last")
        pos_freq[i] = pos_freq[i]*(1-0.022)
    elif i >=5:
        pos_freq[i] = pos_freq[i]*(1-i*0.004)

plt.xticks(pos_freq,freq_leg_str)
x8Hz = 0.1315
x30Hz = 0.737
col = "black"
ls = "--"
lw = 0.7
extent= [0, 1, 0, 1]
axs.axvline(x=x8Hz,color=col,ls=ls,lw=lw)
axs.axvline(x=x30Hz,color=col,ls=ls,lw=lw)
plt.yticks(np.linspace(1/(len(elecs)*2.5),1-1/(len(elecs)*2.5),len(elecs)),elecs.iloc[::-1])
for elecPos in [0.107,0.286,0.428,0.608,0.75,0.9293]:
    axs.axhline(y=elecPos,color="dimgray",lw=0.25)
#img = axs.imshow(df_subset, extent=[0, 1, 0, 1],cmap=cmap, aspect='auto',interpolation='none',vmin=-0.14,vmax=0.14,label="agency") 
img = axs.imshow(masked_perfag, extent=[0, 1, 0, 1],cmap=cmap, aspect='auto',interpolation='none',vmin=vmin,vmax=vmax,label="agency") 
#img = axs.imshow(df_pval_subset_perfag, extent=[0, 1, 0, 1],cmap="Blues", aspect='auto',interpolation='none',vmin=0,vmax=1,label="agency") 
#♦img = axs.imshow(corrected_pval, extent=[0, 1, 0, 1],cmap=cmap, aspect='auto',vmin=vmin,vmax=vmax,label="agency",interpolation='none')
#contour = axs.contour(df_pval_subset_perfag,levels=[0.01,0.05,0.08], colors='black', linewidths=1, extent=extent, corner_mask='legacy', origin='upper',linestyles=["solid","dashed","dotted"],extend="min") 

fig.colorbar(img, location = 'right')
plt.show()

vmax = 0.08
vmin = -vmax
fig, axs = plt.subplots(1,5, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
plot_topomapV3(3,7,df_subset_perfag,vmin,vmax,my_cmap,axs,0)
plot_topomapV3(8,12,df_subset_perfag,vmin,vmax,my_cmap,axs,1)
plot_topomapV3(12,15,df_subset_perfag,vmin,vmax,my_cmap,axs,2)
plot_topomapV3(13,20,df_subset_perfag,vmin,vmax,my_cmap,axs,3)
im = plot_topomapV3(21,30,df_subset_perfag,vmin,vmax,my_cmap,axs,4)
fig.colorbar(im, location = 'right', shrink=0.5)

plt.show()



#======LES MAIN EFFECTS=======


pval_perf = pd.read_csv(path+"df_global_pval_v3_2runs_NFperfAgTogether_perf.csv").iloc[:, 1:]
pval_ag = pd.read_csv(path+"df_global_pval_v3_2runs_NFperfAgTogether_ag.csv").iloc[:, 1:]

pval_perf = pval_perf.to_numpy()
pval_ag = pval_ag.to_numpy()

est_perf = pd.read_csv(path+"df_global_estimate_v3_NFperfAgTogether_perf.csv").iloc[:, 1:]
est_ag = pd.read_csv(path+"df_global_estimate_v3_NFperfAgTogether_ag.csv").iloc[:, 1:]

est_perf = est_perf.to_numpy()
est_ag = est_ag.to_numpy()


fmin = 3
fmax = 40
len_to_keep = fmax - fmin
df_subset = est_perf[:,0:len_to_keep+1]
df_subset_agency = est_ag[:,0:len_to_keep+1]
df_pval_subset = pval_perf[:,0:len_to_keep+1]
df_pval_subset_agency = pval_ag[:,0:len_to_keep+1]


cmap = "RdBu_r"


pvalue = 0.025
p_ag_corr=  mne.stats.fdr_correction(df_pval_subset_agency)[1]
p_perf_corr=  mne.stats.fdr_correction(df_pval_subset)[1]


masked_ag = np.ma.masked_where((df_pval_subset_agency > pvalue) , df_subset_agency)
masked_perf = np.ma.masked_where((df_pval_subset > pvalue) , df_subset)


elecs = elec_leg 
fig, axs = plt.subplots(1,1, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
pos_freq = np.linspace(0.015,0.985,len(freq_leg))
for i in range(len(pos_freq)):
    print(i)
    if i<3:
        pos_freq[i] = pos_freq[i]*(1-i*0.014)
    elif i==3:
        pos_freq[i] = pos_freq[i]*(1-i*0.012)
    elif i ==4:
        pos_freq[i] = pos_freq[i]*(1-i*0.008)
    elif i==len(pos_freq)-1:
        print("last")
        pos_freq[i] = pos_freq[i]*(1-0.022)
    elif i >=5:
        pos_freq[i] = pos_freq[i]*(1-i*0.004)

plt.xticks(pos_freq,freq_leg_str)
x8Hz = 0.1315
x30Hz = 0.737
col = "black"
ls = "--"
lw = 0.7
axs.axvline(x=x8Hz,color=col,ls=ls,lw=lw)
axs.axvline(x=x30Hz,color=col,ls=ls,lw=lw)
plt.yticks(np.linspace(1/(len(elecs)*2.5),1-1/(len(elecs)*2.5),len(elecs)),elecs.iloc[::-1])
for elecPos in [0.107,0.286,0.428,0.608,0.75,0.9293]:
    axs.axhline(y=elecPos,color="dimgray",lw=0.25)
img = axs.imshow(df_subset, extent=[0, 1, 0, 1],cmap=cmap, aspect='auto',interpolation='none',vmin=-0.45,vmax=0.45,label="agency") 
contour = axs.contour(df_pval_subset,levels=[0.01,0.05], colors='black', linewidths=1, extent=extent, corner_mask='legacy', origin='upper',linestyles=["solid","dashed"],extend="min") 

fig.colorbar(img, location = 'right')

plt.show()

vmax = 0.37
vmin = -vmax
fig, axs = plt.subplots(1,5, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
plot_topomapV3(3,7,df_subset,vmin,vmax,my_cmap,axs,0)
plot_topomapV3(8,12,df_subset,vmin,vmax,my_cmap,axs,1)
plot_topomapV3(12,15,df_subset,vmin,vmax,my_cmap,axs,2)
plot_topomapV3(13,20,df_subset,vmin,vmax,my_cmap,axs,3)
im = plot_topomapV3(21,30,df_subset,vmin,vmax,my_cmap,axs,4)
fig.colorbar(im, location = 'right', shrink=0.5)


elecs = elec_leg 
fig, axs = plt.subplots(1,1, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
pos_freq = np.linspace(0.015,0.985,len(freq_leg))
for i in range(len(pos_freq)):
    print(i)
    if i<3:
        pos_freq[i] = pos_freq[i]*(1-i*0.014)
    elif i==3:
        pos_freq[i] = pos_freq[i]*(1-i*0.012)
    elif i ==4:
        pos_freq[i] = pos_freq[i]*(1-i*0.008)
    elif i==len(pos_freq)-1:
        print("last")
        pos_freq[i] = pos_freq[i]*(1-0.022)
    elif i >=5:
        pos_freq[i] = pos_freq[i]*(1-i*0.004)

plt.xticks(pos_freq,freq_leg_str)
x8Hz = 0.1315
x30Hz = 0.737
col = "black"
ls = "--"
lw = 0.7
axs.axvline(x=x8Hz,color=col,ls=ls,lw=lw)
axs.axvline(x=x30Hz,color=col,ls=ls,lw=lw)
plt.yticks(np.linspace(1/(len(elecs)*2.5),1-1/(len(elecs)*2.5),len(elecs)),elecs.iloc[::-1])
for elecPos in [0.107,0.286,0.428,0.608,0.75,0.9293]:
    axs.axhline(y=elecPos,color="dimgray",lw=0.25)
img = axs.imshow(df_subset_agency, extent=[0, 1, 0, 1],cmap=cmap, aspect='auto',interpolation='none',vmin=-0.02,vmax=0.02,label="agency") 
contour = axs.contour(df_pval_subset_agency,levels=[0.01,0.05], colors='black', linewidths=1, extent=extent, corner_mask='legacy', origin='upper',linestyles=["solid","dashed"],extend="min") 
fig.colorbar(img, location = 'right')
plt.show()

cmap = "RdBu_r"
my_cmap = discrete_cmap(13, cmap)

vmax = 0.017
vmin = -vmax
fig, axs = plt.subplots(1,5, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
plot_topomapV3(3,7,df_subset_agency,vmin,vmax,my_cmap,axs,0)
plot_topomapV3(8,12,df_subset_agency,vmin,vmax,my_cmap,axs,1)
plot_topomapV3(12,15,df_subset_agency,vmin,vmax,my_cmap,axs,2)
plot_topomapV3(13,20,df_subset_agency,vmin,vmax,my_cmap,axs,3)
im = plot_topomapV3(21,30,df_subset_agency,vmin,vmax,my_cmap,axs,4)
fig.colorbar(im, location = 'right', shrink=0.5)


#LES DEUX MAIN A COTE
elecs = elec_leg 
fig, axs = plt.subplots(1,2, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
freq_leg = np.arange(3,40,4)
freq_leg_str =[str(f) for f in freq_leg]
pos_freq = np.linspace(0.015,0.985,len(freq_leg))
for i in range(len(pos_freq)):
    print(i)
    if i<3:
        pos_freq[i] = pos_freq[i]*(1-i*0.014)
    elif i==3:
        pos_freq[i] = pos_freq[i]*(1-i*0.012)
    elif i ==4:
        pos_freq[i] = pos_freq[i]*(1-i*0.008)
    elif i==len(pos_freq)-1:
        print("last")
        pos_freq[i] = pos_freq[i]*(1-0.022)
    elif i >=5:
        pos_freq[i] = pos_freq[i]*(1-i*0.004)

plt.xticks(pos_freq,freq_leg_str)
x8Hz = 0.1315
x30Hz = 0.737
col = "black"
ls = "--"
lw = 0.7
extent= [0, 1, 0, 1]
for ax in axs.flat:
    ax.axvline(x=x8Hz,color=col,ls=ls,lw=lw)
    ax.axvline(x=x30Hz,color=col,ls=ls,lw=lw)
plt.yticks(np.linspace(1/(len(elecs)*2.5),1-1/(len(elecs)*2.5),len(elecs)),elecs.iloc[::-1])
for ax in axs.flat:
    for elecPos in [0.107,0.286,0.428,0.608,0.75,0.9293]:
        ax.axhline(y=elecPos,color="dimgray",lw=0.25)
plt.yticks(np.linspace(1/(len(elecs)*2.5),1-1/(len(elecs)*2.5),len(elecs)),elecs.iloc[::-1])
img = axs[0].imshow(masked_perf, extent=extent,cmap=cmap, aspect='auto',interpolation='none',vmin=-0.35,vmax=0.35,label="agency") 
#fig.colorbar(img, location = 'left')
img = axs[1].imshow(masked_ag, extent=extent,cmap=cmap, aspect='auto',interpolation='none',vmin=-0.02,vmax=0.02,label="perf") 
fig.colorbar(img, location = 'right')