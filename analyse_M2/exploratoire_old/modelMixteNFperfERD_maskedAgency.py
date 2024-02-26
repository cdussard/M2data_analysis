# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 15:22:13 2023

@author: claire.dussard
"""


import pandas as pd
import mne
import numpy as np
import imagesc
import matplotlib.pyplot as plt
import scipy

# Path to the directory containing the CSV files
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/fig_brian/correlERD_NFperf_colorgrid/"


# Read the CSV files
df_global = pd.read_csv(path + "df_global_estimateFBseul_v3_2runs_NFperf.csv", header=None, delimiter=",").iloc[1:, 1:].values
df_global_pval = pd.read_csv(path + "df_global_pvalFBseul_v3_2runs_NFperf.csv", header=None, delimiter=",").iloc[1:, 1:].values

path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/fig_brian/correlERD_agency_colorgrid/"
df_global_agency = pd.read_csv(path + "df_global_estimateFBseul_v3_2runs.csv", header=None, delimiter=",").iloc[1:, 1:].values
df_global_pval_agency = pd.read_csv(path + "df_global_pvalFBseul_v3_2runs.csv", header=None, delimiter=",").iloc[1:, 1:].values

df_global = df_global.astype(float)
df_global_agency = df_global_agency.astype(float)
df_global_pval = df_global_pval.astype(float)
df_global_pval_agency = df_global_pval_agency.astype(float)

#subset from fmin to fmax
fmin = 3
fmax = 40
len_to_keep = fmax - fmin
df_subset = df_global[:,0:len_to_keep+1]
df_subset_agency = df_global_agency[:,0:len_to_keep+1]
df_pval_subset = df_global_pval[:,0:len_to_keep+1]
df_pval_subset_agency = df_global_pval_agency[:,0:len_to_keep+1]
print(df_pval_subset.min())
vmin = -0.35
vmax = -vmin
cmap = "RdBu_r"

#FDR correction

corrected_pval = mne.stats.fdr_correction(df_pval_subset)[1]
print(corrected_pval.min())

pvalue = 0.05
masked_global = np.ma.masked_where((corrected_pval > pvalue) , df_subset)
masked_global.data
df_subset[corrected_pval < pvalue]

path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/rdom_scriptsData/allElecFreq_VSZero/versionJuin2023_elecFixed/"
elec_leg = pd.read_csv(path+"dcohen_mainIllusion.csv").iloc[:, 0]


#apres correction FDR
cmap = "RdBu_r"
vmin =-0.03 #-0.35
vmax =0.03 #0.35#0.01
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
img = axs.imshow(df_subset_agency, extent=[0, 1, 0, 1],cmap=cmap, aspect='auto',interpolation='none',vmin=vmin,vmax=vmax,label="agency") 
#img = axs.imshow(df_subset, extent=[0, 1, 0, 1],cmap=cmap, aspect='auto',interpolation='none',vmin=vmin,vmax=vmax,label="agency") 
#♦img = axs.imshow(corrected_pval, extent=[0, 1, 0, 1],cmap=cmap, aspect='auto',vmin=vmin,vmax=vmax,label="agency",interpolation='none')
fig.colorbar(img, location = 'right')

# Plot the significant areas with black contour
#axs.contour(corrected_pval, levels=[0.01,0.05], colors='black', linewidths=0.7,extent=[0, 1, 0, 1],vmin=0,vmax = 0.05)
#data = df_pval_subset
contour = axs.contour(dataperf,levels=[0.01], colors='black', linewidths=0.6, extent=[0, 1, 0, 1])
contour = axs.contour(dataagency,levels=[0.01], colors='blue', linewidths=0.6, extent=[0, 1, 0, 1], corner_mask='legacy')
plt.show()

# Plot the significant areas with black contour
#axs.contour(corrected_pval, levels=[0.01,0.05], colors='black', linewidths=0.7,extent=[0, 1, 0, 1],vmin=0,vmax = 0.05)
#data = df_pval_subset
data = scipy.ndimage.zoom(df_pval_subset, 2)
contour = axs.contour(data,levels=[0.05], colors='black', linewidths=0.6, extent=[0, 1, 0, 1])
contour = axs.contour(data,levels=[0.01], colors='blue', linewidths=0.6, extent=[0, 1, 0, 1])
plt.show()

plt.savefig('try_contours_agency_smooth2.png', transparent=True)


#data = scipy.ndimage.zoom(df_pval_subset, 2)



contour = axs.contour(df_pval_subset,levels=[0.01], colors='blue', linewidths=0.6, extent=[0, 1, 0, 1])
contour = axs.contour(data,levels=[0.01], colors='blue', linewidths=0.6, extent=[0, 1, 0, 1])
# Adjust the aspect ratio of the contours
#axs.set_aspect('auto')
plt.show()

plt.savefig('try_contours_2colors_smooth2.png', transparent=True)



# SI 0N  FAIT UN CONTOUR POUR SIGNIF PERF ET UN CONTOUR POUR SIGNIF AGENCY 
#contour = axs.contour(df_subset, levels=contour_levels, colors='black', linewidths=0.7, extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)


fig, axs = plt.subplots(1,1, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
img = axs.imshow(df_pval_subset, extent=[0, 1, 0, 1],cmap="Blues", aspect='auto',interpolation='none',vmin=0,vmax=vmax,label="agency") 
fig.colorbar(img, location = 'right')
contour = axs.contour(df_pval_subset,levels=[0.01], colors='black', linewidths=0.6, extent=[0, 1, 0, 1], corner_mask='legacy')

contour = axs.contour(df_pval_subset_agency,levels=[0.01], colors='blue', linewidths=0.6, extent=[0, 1, 0, 1], corner_mask='legacy')
plt.show()
plt.savefig('try_contours_blackPerf_blueAgency.png', transparent=True)

#avec smoothing
dataperf =df_pval_subset# scipy.ndimage.zoom(df_pval_subset, 2)
dataagency = df_pval_subset_agency#scipy.ndimage.zoom(df_pval_subset_agency, 2)
fig, axs = plt.subplots(1,1, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
contour = axs.contour(dataperf,levels=[0.01], colors='black', linewidths=1.5, extent=[0, 1, 0, 1],linestyles="dashed")
#contour = axs.contour(dataagency,levels=[0.01], colors='blue', linewidths=1.5, extent=[0, 1, 0, 1],linestyles="dashed")
plt.show()
plt.savefig('try_contours_noirPerf_0.01.png', transparent=True)
plt.savefig('try_contours_blueAgency_0.01.png', transparent=True)
plt.savefig('try_contours_blackPerf_blueAgency_0.01.png', transparent=True)

plt.savefig('try_contours_blackPerf_blueAgency_smooth2.png', transparent=True)


corrected_pval = mne.stats.fdr_correction(df_pval_subset)[1]
corrected_pval_agency = mne.stats.fdr_correction(df_pval_subset_agency)[1]
fig, axs = plt.subplots(1,1, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
contour = axs.contour(corrected_pval,levels=[0.05], colors='black', linewidths=1.5, extent=[0, 1, 0, 1])
contour = axs.contour(corrected_pval_agency,levels=[0.05], colors='blue', linewidths=1.5, extent=[0, 1, 0, 1],linestyles="dashed", corner_mask='legacy')
plt.show()
plt.savefig('try_contours_blackPerf_blueAgency_0.05FDRcorr.png', transparent=True)


#==================================================
#si on fait avec en fond les données d'ERD pures (PB = on perd le sens de la correl)
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/rdom_scriptsData/allElecFreq_VSZero/versionJuin2023_elecFixed/"

pend = pd.read_csv(path+"dcohen_mainIllusion.csv").iloc[:, 1:]
main = pd.read_csv(path+"dcohen_main.csv").iloc[:, 1:]
mIll = pd.read_csv(path+"dcohen_pend.csv").iloc[:, 1:]

pend = pend.to_numpy()
main = main.to_numpy()
mIll = mIll.to_numpy()

moy_ERD = (pend + main+ mIll) /3

fmin = 3
fmax = 40
len_to_keep = fmax - fmin
df_subset = moy_ERD[:,0:len_to_keep+1]

print(df_pval_subset.min())
vmin = 0
vmax = 1.7
cmap = "Blues"

fig, axs = plt.subplots(1,1, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
axs.imshow(-moy_ERD, extent=[0, 1, 0, 1],cmap="Blues", aspect='auto',interpolation='none',vmin=vmin,vmax=vmax)

#•====


fig, axs = plt.subplots(1,1, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
gridspec_kw={'width_ratios': [1,1,1],
                           'height_ratios': [1],
                       'wspace': 0.05,#constrained_layout=True
                       'hspace': 0.05}

img = axs.imshow(-df_subset, extent=[0, 1, 0, 1],cmap="Blues", aspect='auto',interpolation='none',vmin=vmin,vmax=vmax,label="moyFB")
axs.text(0.12, 1.02, 'D cohen ERD')

fig.colorbar(img, location = 'right')
elecs = elec_leg 
#plt.subplots_adjust(wspace=0.2, hspace=0.05)
freq_leg = np.arange(3,40,4)
freq_leg_str =[str(f) for f in freq_leg]
plt.xticks(np.linspace(0.01,0.96,len(freq_leg_str)),freq_leg_str)
# x8Hz = 0.128
# x30Hz = 0.6935
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
#plt.tight_layout(pad=0.04) 
plt.show()


#LIGNE POINTILLE SI MODELE PERF 
#LIGNE SOLIDE SI MODELE AGENCY 
# Path to the directory containing the CSV files
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/fig_brian/correlERD_NFperf_colorgrid/"


# Read the CSV files
df_global = pd.read_csv(path + "df_global_estimateFBseul_v3_2runs_NFperf.csv", header=None, delimiter=",").iloc[1:, 1:].values
df_global_pval = pd.read_csv(path + "df_global_pvalFBseul_v3_2runs_NFperf.csv", header=None, delimiter=",").iloc[1:, 1:].values

path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/fig_brian/correlERD_agency_colorgrid/"
df_global_agency = pd.read_csv(path + "df_global_estimateFBseul_v3_2runs.csv", header=None, delimiter=",").iloc[1:, 1:].values
df_global_pval_agency = pd.read_csv(path + "df_global_pvalFBseul_v3_2runs.csv", header=None, delimiter=",").iloc[1:, 1:].values

df_global = df_global.astype(float)
df_global_agency = df_global_agency.astype(float)
df_global_pval = df_global_pval.astype(float)
df_global_pval_agency = df_global_pval_agency.astype(float)

#subset from fmin to fmax
fmin = 3
fmax = 40
len_to_keep = fmax - fmin
df_subset = df_global[:,0:len_to_keep+1]
df_subset_agency = df_global_agency[:,0:len_to_keep+1]
df_pval_subset = df_global_pval[:,0:len_to_keep+1]
df_pval_subset_agency = df_global_pval_agency[:,0:len_to_keep+1]
print(df_pval_subset.min())
vmin = -0.35
vmax = -vmin
cmap = "RdBu_r"

dataperf_pos = df_pval_subset.copy()
dataperf_neg = df_pval_subset.copy()
dataagency_neg = df_pval_subset_agency.copy()
dataagency_pos = df_pval_subset_agency.copy()

print(dataperf_pos.min())
count = 0
for i in range(dataperf_pos.shape[0]):
    for j in range(dataperf_pos.shape[1]):
        if df_subset[i,j]<0:#perf data
            dataperf_pos[i,j]=None#pour pas que ça s'affiche
        else:
            count = count + 1
            print("pos")
            dataperf_neg[i,j]=None#pour pas que ça s'affiche
print(count)

print(dataperf_pos.min())
            
for i in range(dataperf_pos.shape[0]):
    for j in range(dataperf_pos.shape[1]):
        if df_subset_agency[i,j]<0:#perf data
            dataagency_pos[i,j]=None#pour pas que ça s'affiche
        else:
            dataagency_neg[i,j]=None#pour pas que ça s'affiche
    

fig, axs = plt.subplots(1,1, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
contour = axs.contour(dataperf_pos,levels=[0.01], colors='green', linewidths=1, extent=[0, 1, 0, 1],linestyles="dashed")#positive perf
contour = axs.contour(dataagency_pos,levels=[0.01], colors='green', linewidths=0.9, extent=[0, 1, 0, 1])#positive agency
contour = axs.contour(dataperf_neg,levels=[0.01], colors='red', linewidths=1, extent=[0, 1, 0, 1],linestyles="dashed")#negative perf
contour = axs.contour(dataagency_neg,levels=[0.01], colors='red', linewidths=0.9, extent=[0, 1, 0, 1], corner_mask='legacy'))#negative agency



def get_contour(p_value_files,p_value_level,saveName,linestyles,colors,linewidth):
    fig, axs = plt.subplots(1,1, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
    for (p_value_file,color,linestyle) in zip(p_value_files,colors,linestyles):
        contour = axs.contour(p_value_file,levels=[p_value_level], colors=color, linewidths=linewidth, extent=[0, 1, 0, 1],linestyles=linestyle,origin="upper")#positive perf
    if saveName!=None:
        plt.show()
        plt.savefig(saveName+".png", transparent=True)
get_contour([dataperf_neg,dataagency_neg],0.01,None,["dashed","solid"],["black","blue"],1.5)

    