# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:07:42 2023

@author: claire.dussard
"""

import pandas as pd
import mne
import numpy as np
import imagesc

# Path to the directory containing the CSV files
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/fig_brian/correlERD_NFperf_colorgrid/"


# Read the CSV files
df_global = pd.read_csv(path + "df_global_estimateFBseul_v3_2runs_NFperf.csv", header=None, delimiter=",").iloc[1:, 1:].values
df_global = pd.read_csv(path + "df_global_estimateFBseul_v3_2runs_ERD.csv", header=None, delimiter=",").iloc[1:, 1:].values
#df_global = pd.read_csv(path + "df_global_estimateFBseul_v3_2runs_ERD_rd.csv", header=None, delimiter=",").iloc[1:, 1:].values
#df_global = pd.read_csv(path + "df_global_estimateFBseul_v3_2runs_ERDAg_rd.csv", header=None, delimiter=",").iloc[1:, 1:].values
df_global_pval2 = pd.read_csv(path + "df_global_pvalFBseul_v3_2runs_NFperf.csv", header=None, delimiter=",").iloc[1:, 1:].values
df_global_pval = pd.read_csv(path + "df_global_pvalFBseul_v3_2runs_NFperf_anova2.csv", header=None, delimiter=",").iloc[1:, 1:].values
df_global_pval = pd.read_csv(path + "df_global_pvalFBseul_v3_2runs_ERD.csv", header=None, delimiter=",").iloc[1:, 1:].values
#df_global_pval = pd.read_csv(path + "df_global_pvalFBseul_v3_2runs_ERD_rd.csv", header=None, delimiter=",").iloc[1:, 1:].values
#df_global_pval = pd.read_csv(path + "df_global_pvalFBseul_v3_2runs_ERDAg_rd.csv", header=None, delimiter=",").iloc[1:, 1:].values

df_global = df_global.astype(float)
df_global_pval = df_global_pval.astype(float)
df_global_pval2 = df_global_pval2.astype(float)

#subset from fmin to fmax
fmin = 3
fmax = 40
len_to_keep = fmax - fmin
df_subset = df_global[:,0:len_to_keep+1]
df_pval_subset = df_global_pval[:,0:len_to_keep+1]
df_pval_subset2 = df_global_pval2[:,0:len_to_keep+1]
print(df_pval_subset.min())
vmin = -0.55#-0.35
vmax = -vmin
cmap = "RdBu_r"

#FDR correction

corrected_pval = mne.stats.fdr_correction(df_pval_subset)[1]
corrected_pval2 = mne.stats.fdr_correction(df_pval_subset2)[1]
print(corrected_pval.min())

pvalue = 0.05

masked_global = np.ma.masked_where((corrected_pval > pvalue) , df_subset)
masked_global2 = np.ma.masked_where((corrected_pval2 > pvalue) , df_subset)
masked_global.data
df_subset[corrected_pval < pvalue]


import matplotlib.pyplot as plt
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/rdom_scriptsData/allElecFreq_VSZero/versionJuin2023_elecFixed/"
elec_leg = pd.read_csv(path+"dcohen_mainIllusion.csv").iloc[:, 0]

#apres correction FDR
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
axs.axvline(x=x8Hz,color=col,ls=ls,lw=lw)
axs.axvline(x=x30Hz,color=col,ls=ls,lw=lw)
plt.yticks(np.linspace(1/(len(elecs)*2.5),1-1/(len(elecs)*2.5),len(elecs)),elecs.iloc[::-1])
for elecPos in [0.107,0.286,0.428,0.608,0.75,0.9293]:
    axs.axhline(y=elecPos,color="dimgray",lw=0.25)
img = axs.imshow(masked_global, extent=[0, 1, 0, 1],cmap=cmap, aspect='auto',interpolation='none',vmin=vmin,vmax=vmax,label="agency") 
fig.colorbar(img, location = 'right')
plt.show()

vmax = 0.4
vmin = -vmax
fig, axs = plt.subplots(1,5, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
plot_topomapV3(3,7,df_subset,vmin,vmax,my_cmap,axs,0)
plot_topomapV3(8,12,df_subset,vmin,vmax,my_cmap,axs,1)
plot_topomapV3(12,15,df_subset,vmin,vmax,my_cmap,axs,2)
plot_topomapV3(13,20,df_subset,vmin,vmax,my_cmap,axs,3)
im = plot_topomapV3(21,30,df_subset,vmin,vmax,my_cmap,axs,4)
fig.colorbar(im, location = 'right', shrink=0.5)

plot_topomapV2(8,30,df_subset,vmin,vmax,my_cmap)

#==============avant fdr====================
pvalue = 0.05
masked_global = np.ma.masked_where((df_pval_subset > pvalue) , df_subset)
masked_global.data
df_subset[df_pval_subset < pvalue]


import matplotlib.pyplot as plt
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/rdom_scriptsData/allElecFreq_VSZero/versionJuin2023_elecFixed/"
elec_leg = pd.read_csv(path+"dcohen_mainIllusion.csv").iloc[:, 0]

fig, axs = plt.subplots(1,1, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
freq_leg = np.arange(3,40,4)
freq_leg_str =[str(f) for f in freq_leg]
plt.xticks(np.linspace(0.01,0.99,len(freq_leg)),freq_leg_str)
axs.axvline(x=x8Hz,color=col,ls=ls,lw=lw)
axs.axvline(x=x30Hz,color=col,ls=ls,lw=lw)
plt.yticks(np.linspace(1/(len(elecs)*2.5),1-1/(len(elecs)*2.5),len(elecs)),elecs.iloc[::-1])
for elecPos in [0.107,0.286,0.428,0.608,0.75,0.9293]:
    axs.axhline(y=elecPos,color="dimgray",lw=0.25)
img = axs.imshow(masked_global, extent=[0, 1, 0, 1],cmap=cmap, aspect='auto',interpolation='none',vmin=vmin,vmax=vmax,label="agency") 
fig.colorbar(img, location = 'right')


#========== UNCORRECTED =========================
fig, axs = plt.subplots(1,1, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
freq_leg = np.arange(3,40,4)
freq_leg_str =[str(f) for f in freq_leg]
plt.xticks(np.linspace(0,1,len(freq_leg)),freq_leg_str)
axs.axvline(x=x8Hz,color=col,ls=ls,lw=lw)
axs.axvline(x=x30Hz,color=col,ls=ls,lw=lw)
plt.yticks(np.linspace(1/(len(elecs)*2.5),1-1/(len(elecs)*2.5),len(elecs)),elecs.iloc[::-1])
for elecPos in [0.107,0.286,0.428,0.608,0.75,0.9293]:
    axs.axhline(y=elecPos,color="dimgray",lw=0.25)
img = axs.imshow(df_subset, extent=[0, 1, 0, 1],cmap=cmap, aspect='auto',interpolation='none',vmin=vmin,vmax=vmax,label="agency") 
fig.colorbar(img, location = 'right')
plt.show()


#ajouter masque avec differentes signif
masked_global_pval = np.ma.masked_where((corrected_pval > 0.05) , corrected_pval)
masked_global_pval2 = np.ma.masked_where((corrected_pval > 0.01) , corrected_pval)
masked_global_pval3 = np.ma.masked_where((corrected_pval > 0.001) , corrected_pval)


plt.pcolor(masked_global_pval, cmap='RdBu_r',edgecolors="black", linewidths=1, facecolors='none')
plt.pcolor(masked_global_pval2, cmap='RdBu_r',edgecolors="green", linewidths=1, facecolors='none')
plt.pcolor(masked_global_pval3, cmap='RdBu_r',edgecolors="blue", linewidths=1, facecolors='none')
plt.savefig('mask_noir0.05_vert0.01_bleu0.001_avecCorrection.png', transparent=True)

masked_global_pval = np.ma.masked_where((df_pval_subset > 0.05) , df_pval_subset)
masked_global_pval2 = np.ma.masked_where((df_pval_subset > 0.01) , df_pval_subset)
masked_global_pval3 = np.ma.masked_where((df_pval_subset > 0.001) , df_pval_subset)
masked_global_pval4 = np.ma.masked_where((df_pval_subset > 0.0001) , df_pval_subset)


#plt.pcolor(masked_global_pval, cmap='RdBu_r',edgecolors="black", linewidths=1, facecolors='none')
plt.pcolor(masked_global_pval2, cmap='RdBu_r',edgecolors="green", linewidths=1, facecolors='none')
plt.pcolor(masked_global_pval3, cmap='RdBu_r',edgecolors="blue", linewidths=1, facecolors='none')
plt.pcolor(masked_global_pval4, cmap='RdBu_r',edgecolors="red", linewidths=1, facecolors='none')


plt.savefig('mask_vert0.01_bleu0.001_rouge0.0001_sansCorrection.png', transparent=True)

plt.savefig('mask_noir0.05_vert0.01_bleu0.001_sansCorrection.png', transparent=True)