# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:23:17 2023

@author: claire.dussard
"""

import mne

from functions.load_savedData import *
from handleData_subject import createSujetsData
from functions.load_savedData import *
import numpy as np
import os
import pandas as pd

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()

#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

liste_tfrPendule = load_tfr_data_windows(liste_rawPathPendule,"",True)
liste_tfrMain = load_tfr_data_windows(liste_rawPathMain,"",True)
liste_tfrMainIllusion = load_tfr_data_windows(liste_rawPathMainIllusion,"",True)

def copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion):
#avoid having to reload from scratch after every ANOVA (instances modified in place)
    liste_tfr_pendule = []
    liste_tfr_main = []
    liste_tfr_mainIllusion = []
    for tfr_p,tfr_m,tfr_mi in zip(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion):
        liste_tfr_pendule.append(tfr_p.copy())
        liste_tfr_main.append(tfr_m.copy())
        liste_tfr_mainIllusion.append(tfr_mi.copy())
    return liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion

def data_freq_tTest_perm(elec,fmin,fmax,tmin,tmax,liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule):
    mode_baseline = 'logratio'
    n_sujets = len(liste_tfr_pendule)
    baseline = (-3,-1)
    #compute baseline (first because after we crop time)
    for tfr_m,tfr_mi,tfr_p in zip(liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule):
        tfr_m.apply_baseline(baseline=baseline, mode=mode_baseline, verbose=None)
        tfr_mi.apply_baseline(baseline=baseline, mode=mode_baseline, verbose=None)
        tfr_p.apply_baseline(baseline=baseline, mode=mode_baseline, verbose=None)
    #crop time & frequency
    for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
        tfr_mainI.crop(tmin = tmin,tmax=tmax,fmin = fmin,fmax = fmax)
        tfr_main.crop(tmin = tmin,tmax=tmax,fmin = fmin,fmax = fmax)
        tfr_pendule.crop(tmin = tmin,tmax=tmax,fmin = fmin,fmax = fmax)
    #subset electrode
    for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
        tfr_mainI.pick_channels([elec])
        tfr_main.pick_channels([elec])
        tfr_pendule.pick_channels([elec])
    #create ANOVA table "faire evoluer pour plusieurs elecs a la fois
    tableau_mainPendule = np.zeros(shape=(n_sujets,fmax-fmin+1))#23 = nb sujets
    tableau_mainMainIllusion = np.zeros(shape=(n_sujets,fmax-fmin+1))
    tableau_main = np.zeros(shape=(n_sujets,fmax-fmin+1))
    tableau_pendule = np.zeros(shape=(n_sujets,fmax-fmin+1))
    tableau_mainIllusion = np.zeros(shape=(n_sujets,fmax-fmin+1))
    for i in range(n_sujets):#sujets
        print("sujet"+str(i))
        #ecraser forme electrodes
        liste_tfr_pendule[i].data = np.mean(liste_tfr_pendule[i].data,axis=0)
        liste_tfr_main[i].data = np.mean(liste_tfr_main[i].data,axis=0)
        liste_tfr_mainIllusion[i].data = np.mean(liste_tfr_mainIllusion[i].data,axis=0)
        #pool time
        powerFreq_pendule = np.median(liste_tfr_pendule[i].data,axis=1)#OK donc dim = freq x time
        powerFreq_main = np.median(liste_tfr_main[i].data,axis=1)
        powerFreq_mainI = np.median(liste_tfr_mainIllusion[i].data,axis=1)
        print(powerFreq_main)
        mainMoinsPendule_i = powerFreq_main - powerFreq_pendule
        print(mainMoinsPendule_i)
        mainMoinsMainIllusion_i = powerFreq_main - powerFreq_mainI
        for j in range(fmax-fmin+1):#freq
            print("freq"+str(fmin+j))
            tableau_mainPendule[i][j] = mainMoinsPendule_i[j]
            print(mainMoinsPendule_i[j])
            tableau_mainMainIllusion[i][j] = mainMoinsMainIllusion_i[j]
            tableau_main[i][j] = powerFreq_main[j]
            tableau_pendule[i][j] = powerFreq_pendule[j]
            tableau_mainIllusion[i][j] = powerFreq_mainI[j]
    return tableau_mainPendule,tableau_mainMainIllusion,tableau_main,tableau_pendule,tableau_mainIllusion

obj_channels=["Fp1","Fp2","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8",
"CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","O1","Oz","O2"]
liste_pendule = []
liste_main = []
liste_mainIllusion = []

#reorder channels

for elec in obj_channels:
    print("ELEC  "+elec)
    liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion = copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion)
    tableau_mainPendule,tableau_mainMainIllusion,tableau_main,tableau_pendule,tableau_mainIllusion = data_freq_tTest_perm(elec,3,84,2.5,25.5,liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule)
    liste_mainIllusion.append(tableau_mainIllusion)
    liste_main.append(tableau_main)
    liste_pendule.append(tableau_pendule)
    
    
    
#now get the p values
def get_pvalue_allElec_allFreq(liste_condition,npermut):
    n_sujets = liste_condition[0].shape[0]
    liste_suj_data = []
    for suj in range(n_sujets):
        for elec in range(28):
            print("suj"+str(suj))
            liste_suj_data.append(liste_condition[elec][suj])
    
    fullTableTest_condition = np.zeros(shape=(n_sujets,82*28))
    for i in range(n_sujets):#sujet
        for j in range(28):#electrodes
            fullTableTest_condition[i:i+1,j*82:(j+1)*82] = liste_suj_data[(28*i)+j]
          
    T0, p_values , H0  = mne.stats.permutation_t_test(fullTableTest_condition,npermut)
    significant_freqs = p_values <= 0.05
    print(significant_freqs)
    
    readable_pValue_table = np.zeros(shape=(28,82)) 
    for i in range(28):#elec
        for j in range(82):#freq
            readable_pValue_table[i,j] = p_values[(82*i)+j]    
    return readable_pValue_table

readable_pValue_table_pendule =  get_pvalue_allElec_allFreq(liste_pendule,20000)  
readable_pValue_table_main =  get_pvalue_allElec_allFreq(liste_main,20000)
readable_pValue_table_mainIllusion =  get_pvalue_allElec_allFreq(liste_mainIllusion,20000) 

header_row =  ["Channels\\freq"] + list(np.arange(3,85,1))  # Adding an empty cell for the top-left corner
header_col = ["Channels\\freq"] + obj_channels


# Creating a DataFrame from the data
df_pval_pendule = pd.DataFrame(readable_pValue_table_pendule, index=header_col[1:], columns=header_row[1:])
df_pval_main = pd.DataFrame(readable_pValue_table_main, index=header_col[1:], columns=header_row[1:])
df_pval_mainIllusion = pd.DataFrame(readable_pValue_table_mainIllusion, index=header_col[1:], columns=header_row[1:])

path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/rdom_scriptsData/allElecFreq_VSZero/versionJuin2023_elecFixed/"

df_pval_pendule.to_csv(path+"p_pend.csv")
df_pval_main.to_csv(path+"p_main.csv")
df_pval_mainIllusion.to_csv(path+"p_mainIllusion.csv")

#now get cohen's d

def get_dcohen_allElec_allFreq(liste_condition):
    ndarray = np.zeros(shape=(28,82))
    for elec in range(28):
        mean = np.mean(liste_condition[elec],axis=0)
        print(mean)
        stdev = np.std(liste_condition[elec],axis=0)
        print(stdev)
        dcohen = mean/stdev
        ndarray[elec]=dcohen
            
    return ndarray
d_p = get_dcohen_allElec_allFreq(liste_pendule)
d_m = get_dcohen_allElec_allFreq(liste_main)
d_mi = get_dcohen_allElec_allFreq(liste_mainIllusion)

# Creating a DataFrame from the data
df_d_pendule = pd.DataFrame(d_p, index=header_col[1:], columns=header_row[1:])
df_d_main = pd.DataFrame(d_m, index=header_col[1:], columns=header_row[1:])
df_d_mainIllusion = pd.DataFrame(d_mi, index=header_col[1:], columns=header_row[1:])

df_d_pendule.to_csv(path+"dcohen_pend.csv")
df_d_main.to_csv(path+"dcohen_main.csv")
df_d_mainIllusion.to_csv(path+"dcohen_mainIllusion.csv")


#check what we get
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


import imagesc
imagesc.plot(pend,cmap="Blues")
imagesc.plot(main,cmap="Blues")
imagesc.plot(mIll,cmap="Blues")

raw_signal.plot(block=True)

pvalue = 0.05/3  
masked_p = np.ma.masked_where((p_pend > pvalue) , pend)
masked_m = np.ma.masked_where((p_main > pvalue) , main)
masked_mi = np.ma.masked_where((p_mIll > pvalue) , mIll)

imagesc.plot(-masked_p,cmap="Blues")
imagesc.plot(-masked_m,cmap="Blues")
imagesc.plot(-masked_mi,cmap="Blues")

import matplotlib.pyplot as plt
elec_leg = pd.read_csv(path+"dcohen_mainIllusion.csv").iloc[:, 0]
gridspec_kw={'width_ratios': [1,1,1],
                           'height_ratios': [1],
                       'wspace': 0.05,#constrained_layout=True
                       'hspace': 0.05}
fig, axs = plt.subplots(1,3, sharey=True,sharex=True, figsize=(20, 7),gridspec_kw=gridspec_kw,constrained_layout=True)
vmin = 0.9
vmax = 2.1
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
x30Hz = 0.34
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
raw_signal.plot(block=True)#specifier le x
