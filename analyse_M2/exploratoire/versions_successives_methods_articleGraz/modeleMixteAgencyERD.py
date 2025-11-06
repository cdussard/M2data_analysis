# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:19:54 2023

@author: claire.dussard
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat

from handle_data_subject import * 

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()
listeNumSujetsFinale.pop(1)
listeNumSujetsFinale.pop(3)

#path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/rdom_scriptsData/allElecFreq_VSZero/refait_25/"
# path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/rdom_scriptsData/allElecFreq_VSZero/versionJuin2023_elecFixed/"

# mIll = pd.read_csv(path+"dcohen_mainIllusion.csv")
# main = pd.read_csv(path+"dcohen_main.csv")
# pend = pd.read_csv(path+"dcohen_pend.csv")
# mIll = pd.read_csv(path+"dcohen_mIll.csv",header=None,delimiter=";")
# main = pd.read_csv(path+"dcohen_main.csv",header=None,delimiter=";")
# pend = pd.read_csv(path+"dcohen_pendule.csv",header=None,delimiter=";")

path = "../../../../MATLAB_DATA/"

all_data = ["pendule","main","mainIllusion"]

import numpy as np
import pandas as pd

all_data = ["pendule", "main", "mainIllusion"]
channels = ["Fp1", "Fp2", "F7", "F3", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
            "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2","Fz"]

freqs = np.arange(3, 84, 1)

n_elec = len(channels)
n_freq = len(freqs)

path = "../../../../MATLAB_DATA/"
num_iterations = len(listeNumSujetsFinale) * len(all_data) * n_elec * n_freq
dict_total = np.empty((num_iterations, 5), dtype=object)
index = 0

num_sujets = allSujetsDispo
for suj,num_suj in zip(listeNumSujetsFinale,num_sujets):
    for FB in all_data:
        path_sujet_fb = path + suj + "/" + suj + "-" + FB + "timePooled.mat"
        print(path_sujet_fb)
        data = loadmat(path_sujet_fb)["data"]

        for elec_i in range(n_elec):
            for freq_i in range(n_freq):
                value = data[elec_i, freq_i]
                dict_total[index] = [num_suj, FB, channels[elec_i], freq_i + 3, value]
                index += 1

dict_total = pd.DataFrame(dict_total[:index], columns=["num_sujet", "FB", "elec", "freq", "ERD_value"])
         
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/rdom_scriptsData/allElecFreq_VSZero/versionJuin2023_elecFixed/"
             
dict_total.to_csv(path+"optimized_dataCorrel.csv")      

#PLOT

import pandas as pd
import numpy as np
import imagesc

# Path to the directory containing the CSV files
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/fig_brian/correlERD_agency_colorgrid/V2/"

# Read the CSV files
df_global = pd.read_csv(path + "df_global_estimateFBseul_v2.csv", header=None, delimiter=",").iloc[1:, 1:].values
df_global_pval = pd.read_csv(path + "df_global_pvalFBseul_v2.csv", header=None, delimiter=",").iloc[1:, 1:].values

df_global = df_global.astype(float)
df_global_pval = df_global_pval.astype(float)

# Convert to numeric values and float dtype

pvalue = 0.01/3  
masked_global = np.ma.masked_where((df_global_pval > pvalue) , df_global)
masked_global.data
df_global[df_global_pval < pvalue]

imagesc.plot(-masked_global,cmap="Blues")
imagesc.plot(-df_global,cmap="Blues")



import matplotlib.pyplot as plt


fig, axs = plt.subplots(1,1, sharey=True,sharex=True, figsize=(20, 7),constrained_layout=True)
vmin = 0
vmax = 0.03
img = axs.imshow(-df_global, extent=[0, 1, 0, 1],cmap="Blues", aspect='auto',interpolation='none',vmin=vmin,vmax=vmax,label="agency")
axs.text(0.12, 1.02, 'Agency effect')

fig.colorbar(img, location = 'right')

elecs = elec_leg 
#plt.subplots_adjust(wspace=0.2, hspace=0.05)
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
#plt.tight_layout(pad=0.04) 
raw_signal.plot(block=True)#specifier le x

masked_global

liste_tfrPendule = load_tfr_data_windows(liste_rawPathPendule[0:2],"",True)
obj_channels=["Fp1","Fp2","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8",
"CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","O1","Oz","O2"]

info = mne.create_info(obj_channels,250,"eeg") #
info.set_montage(mne.channels.make_standard_montage('easycap-M1'))


def plot_topomap(fmin,fmax,masked_global,vmin,vmax):
    print(freqs[fmin-3:fmax-2])
    masked_global_freq = masked_global[:,fmin-3:fmax-2]
    mean = np.mean(masked_global_freq,axis=1)
    print(mean.max())
    print(mean.min())
    fig,ax = plt.subplots(ncols=1)
    im,cm   = mne.viz.plot_topomap(-mean,info, axes=ax,show=False,vmin=vmin,vmax=vmax,cmap="Blues",border=0,sphere='eeglab')   
    ax_x_start = 0.95
    ax_x_width = 0.04
    ax_y_start = 0.1
    ax_y_height = 0.9
    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    
    clb = fig.colorbar(im, cax=cbar_ax)
    clb.ax.set_title("unit_label",fontsize=12)
    return fig,mean

vmin=0
vmax=0.026
fig_3_7,mean = plot_topomap(3,7,masked_global,vmin,vmax)
fig_8_12,mean = plot_topomap(8,12,masked_global,vmin,vmax)
fig_12_15,mean = plot_topomap(12,15,masked_global,vmin,vmax)
fig_20_30,mean = plot_topomap(20,30,masked_global,vmin,vmax)
fig_8_30,mean = plot_topomap(8,30,masked_global,vmin,vmax)

vmin=0
vmax=1.4

pvalue = 0.01/3  
masked_p = np.ma.masked_where((p_pend > pvalue) , pend)
masked_m = np.ma.masked_where((p_main > pvalue) , main)
masked_mi = np.ma.masked_where((p_mIll > pvalue) , mIll)

fig_12_15,mean = plot_topomap(12,15,masked_p,vmin,vmax)
fig_12_15,mean = plot_topomap(12,15,masked_m,vmin,vmax)
fig_12_15,mean = plot_topomap(12,15,masked_mi,vmin,vmax)

vmin=0
vmax=1.5
fig,mean = plot_topomap(8,30,masked_p,vmin,vmax)
fig,mean = plot_topomap(8,30,masked_m,vmin,vmax)
fig,mean = plot_topomap(8,30,masked_mi,vmin,vmax)

fig_,mean = plot_topomap(8,12,masked_p,vmin,vmax)
fig_,mean = plot_topomap(8,12,masked_m,vmin,vmax)
fig_,mean = plot_topomap(8,12,masked_mi,vmin,vmax)

# pval_global = pd.to_numeric(mIll.flatten(), errors='coerce').reshape(mIll.shape).astype(float)
# main = pd.to_numeric(main.flatten(), errors='coerce').reshape(main.shape).astype(float)
# pend = pd.to_numeric(pend.flatten(), errors='coerce').reshape(pend.shape).astype(float)

#channels=["Fp1","Fp2","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8",
#"CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","O1","Oz","O2"]

# channels=["Fp1","Fp2","F7","F3","F4","F8","FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8",
# "CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","O1","Oz","O2","Fz"]
# freqs = np.arange(3,84,1)

#REMPLACER CA PAR UNE LECTURE DE LA PREMIERE LIGNE ET DE LA PREMIERE COLONNE

# channels = pd.read_csv(path+"dcohen_mainIllusion.csv").iloc[:, 0].to_list()
# header_df = pd.read_csv(path+"dcohen_mainIllusion.csv", header=0, nrows=0)
# header_list = list(header_df)
# header_list = [int(val) for val in header_list[1:]]
# freqs = header_list

# n_elec = len(channels)
# n_freq = len(freqs)

# i = 0
# dict_total = pd.DataFrame(columns = ["num_sujet","FB","elec","freq","ERD_value"])
# for suj in listeNumSujetsFinale:
#     for FB in all_data:
#         path_sujet_fb = path+suj+"/"+suj+"-"+FB+"timePooled.mat"
#         print(path_sujet_fb)
#         data = loadmat(path_sujet_fb)["data"]
#         for elec_i in range(n_elec):
#            for freq_i in range(n_freq):
#                value = data[elec_i,freq_i]
#                dict_suj = {"num_sujet": allSujetsDispo[i],
#                 "FB": FB,
#                 "elec":channels[elec_i],
#                 "freq":freq_i+3,
#                 "ERD_value":value }
#                print(dict_suj)
#                dict_total = dict_total.append(dict_suj,ignore_index=True)

    # i += 1

# dict_total.to_csv("dictCSV_agencyCorrel_elecFiexed.csv")
    
#versionopti ?=================







        