# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 17:05:17 2024

@author: claire.dussard
"""
import matplotlib.pyplot as plt
import numpy as np
import mne
import pandas as pd

freqs = np.arange(3,84,1)
obj_channels=["Fp1","Fp2","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8",
"CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","O1","Oz","O2"]
reorganize_data_elec = ["Fp1","F7","F3","FC5","FC1","T7","C3","CP5","CP1","P7","P3","O1"
                        "Fz","Cz","Pz","Oz"
                        "Fp2","F4", "F8","FC2","FC6","T8","CP2","CP6","P4","P8","O2"
                       
                ]


info = mne.create_info(obj_channels,250,"eeg")
info.set_montage(mne.channels.make_standard_montage('easycap-M1'))


path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/rdom_scriptsData/allElecFreq_VSZero/versionJuin2023_elecFixed/"
elecs = pd.read_csv(path+"dcohen_mainIllusion.csv").iloc[:, 0]


def plot_fig_elec(data,vmin,vmax,cmap):
    my_cmap = discrete_cmap(13,cmap)
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
    x12Hz = 0.263
    x30Hz = 0.737
    x20Hz = 0.474
    col = "black"
    ls = "--"
    lw = 0.7
    axs.axvline(x=x8Hz,color=col,ls=ls,lw=lw)
    axs.axvline(x=x12Hz,color=col,ls=ls,lw=lw)
    axs.axvline(x=x20Hz,color=col,ls=ls,lw=lw)
    axs.axvline(x=x30Hz,color=col,ls=ls,lw=lw)
    plt.yticks(np.linspace(1/(len(elecs)*2.5),1-1/(len(elecs)*2.5),len(elecs)),elecs.iloc[::-1])
    for elecPos in [0.107,0.286,0.428,0.608,0.75,0.9293]:
        axs.axhline(y=elecPos,color="dimgray",lw=0.25)
    img = axs.imshow(data, extent=[0, 1, 0, 1],cmap=my_cmap, aspect='auto',interpolation='none',vmin=vmin,vmax=vmax,label="agency") 
    fig.colorbar(img, location = 'right',ax=axs)
    return img

def plot_fig_elec_V2(data,vmin,vmax,cmap,fmax,col):#white ou black pr la col
    my_cmap = discrete_cmap(13,cmap)
    fig, axs = plt.subplots(1,1, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
    freq_leg = np.arange(3,fmax+1,1)
    freq_leg_str =[str(f) for f in freq_leg]
    pos_freq = np.linspace(0.015,0.985,len(freq_leg))    
    plt.xticks(pos_freq,freq_leg_str, fontsize=19)
    x20Hz = 0.474
    #col = "black"
    ls = "--"
    lw = 0.7
    if fmax<40:
        x8Hz = 0.1791
        x12Hz = 0.357
        x20Hz = 0.643
    if fmax>30:
        x30Hz = 0.737
        axs.axvline(x=x30Hz,color=col,ls=ls,lw=lw)
    axs.axvline(x=x8Hz,color=col,ls=ls,lw=lw)
    axs.axvline(x=x12Hz,color=col,ls=ls,lw=lw)
    axs.axvline(x=x20Hz,color=col,ls=ls,lw=lw)
    plt.yticks(np.linspace(1/(len(elecs)*2.5),1-1/(len(elecs)*2.5),len(elecs)),elecs.iloc[::-1])
    for elecPos in [0.107,0.286,0.428,0.608,0.75,0.9293]:
        axs.axhline(y=elecPos,color="dimgray",lw=0.25)
    img = axs.imshow(data, extent=[0, 1, 0, 1],cmap=my_cmap, aspect='auto',interpolation='none',vmin=vmin,vmax=vmax,label="agency") 
    #ajouts pr pres a graz
    cb = fig.colorbar(img, location = 'right',ax=axs)
    axs.set_facecolor('#050519')
    fig.patch.set_facecolor('#050519')
    cb.ax.set_facecolor('#050519')
    cb.ax.tick_params(labelcolor='white', labelsize=25)
    return img

def plot_topomapV3(fmin,fmax,masked_global,vmin,vmax,cmap,axs,i):
    my_cmap = discrete_cmap(13,cmap)
    freqs = np.arange(3,84,1)
    #print(freqs[fmin-3:fmax-2])
    masked_global_freq = masked_global[:,fmin-3:fmax-2]
    mean = np.mean(masked_global_freq,axis=1)
    #print(mean.max())
    #print(mean.min())
    im,cm   = mne.viz.plot_topomap(mean,info, axes=axs[i],show=False,vlim=(vmin,vmax),cmap=my_cmap)#,border=0,sphere='eeglab')   
    return im

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def plot_topomapV2(fmin,fmax,masked_global,vmin,vmax,cmap):
    my_cmap = discrete_cmap(13,cmap)
    print(freqs[fmin-3:fmax-2])
    masked_global_freq = masked_global[:,fmin-3:fmax-2]
    mean = np.mean(masked_global_freq,axis=1)
    print(mean.max())
    print(mean.min())
    fig,ax = plt.subplots(ncols=1)
    im,cm   = mne.viz.plot_topomap(mean,info, axes=ax,show=False,vlim=(vmin,vmax),cmap=my_cmap)#,border=0,sphere='eeglab')   
    fig.colorbar(im, location = 'right')
    return fig,mean




