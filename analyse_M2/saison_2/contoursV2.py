# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 18:11:40 2023

@author: claire.dussard
"""
#path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/fig_brian/correlERD_NFperf_colorgrid/"
#df_global_pval = pd.read_csv(path + "df_global_pvalFBseul_v3_2runs_NFperf.csv", header=None, delimiter=",").iloc[1:, 1:].values
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/fig_brian/correlERD_agency_colorgrid/"
#df_global = pd.read_csv(path + "df_global_estimateFBseul_v3_2runs_NFperf.csv", header=None, delimiter=",").iloc[1:, 1:].values
df_global_pval = pd.read_csv(path + "df_global_pvalFBseul_v3_2runs.csv", header=None, delimiter=",").iloc[1:, 1:].values

#df_subset[corrected_pval < pvalue]
df_global = df_global.astype(float)
df_global_pval = df_global_pval.astype(float)
fmin = 3
fmax = 40
len_to_keep = fmax - fmin
df_subset = df_global[:,0:len_to_keep+1]
df_pval_subset = df_global_pval[:,0:len_to_keep+1]
print(df_pval_subset.min())
vmin = -0.35
vmax = -vmin
cmap = "RdBu_r"
pvalue = 0.05


corrected_pval = mne.stats.fdr_correction(df_pval_subset)[1]
print(corrected_pval.min())
masked_global = np.ma.masked_where((corrected_pval > pvalue) , df_subset)
masked_global_pval = np.ma.masked_where((corrected_pval > pvalue) , corrected_pval)
masked_global_pval2 = np.ma.masked_where((corrected_pval > 0.01) , corrected_pval)
#FDR correction

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


plt.pcolor(masked_global_pval, cmap='RdBu_r',edgecolors="black", linewidths=1, facecolors='none')
plt.pcolor(masked_global_pval2, cmap='RdBu_r',edgecolors="green", linewidths=1, facecolors='none')


plt.savefig('plot2_agency_severe.png', transparent=True)
#ouvrir dans ppt et retourner verticalement

