# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 20:58:37 2024

@author: claire.dussard
"""


import os 
import seaborn as sns
import pathlib
from handleData_subject import createSujetsData
from functions.load_savedData import *
from functions.preprocessData_eogRefait import *
import numpy as np 
import mne

nom_essai = "4"
essaisFeedbackSeul = ["pas_enregistre","sujet jeté",
"4","4","sujet jeté","4","4","4","4","MISSING","4","4",
"4","4","4","4","4","4","4","4-b","4","4","4","4","4","4"]

# essaisFeedbackSeul = [nom_essai for i in range(25)]

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()
sujetsPb = [0,9]
for sujetpb in sujetsPb:
    allSujetsDispo.remove(sujetpb)
liste_rawPathEffetFBseul = createListeCheminsSignaux(essaisFeedbackSeul,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
# plot les NFB a cote des rest
liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)


liste_power_sujets = load_tfr_data_windows(liste_rawPath_rawMIalone,"",True)

def copy_tfrs(liste_tfrMain):
#avoid having to reload from scratch after every ANOVA (instances modified in place)
    liste_tfr_main = []
    for tfr_m in liste_tfrMain:
        liste_tfr_main.append(tfr_m.copy())
    return liste_tfr_main


def data_freq_tTest_perm(elec,fmin,fmax,tmin,tmax,liste_tfr_main):
    mode_baseline = 'logratio'
    n_sujets = len(liste_tfr_main)
    baseline = (-4,-1.5)
    #compute baseline (first because after we crop time)
     #on a deux epochs : mean

    for tfr_m in liste_tfr_main:
        print(tfr_m)
        tfr_m.apply_baseline(baseline=baseline, mode=mode_baseline, verbose=None)

    #crop time & frequency
    for tfr_main in liste_tfr_main:
        tfr_main.copy().crop(tmin = tmin,tmax=tmax,fmin = fmin,fmax = fmax)
    print("here")
    #subset electrode
    for tfr_main in liste_tfr_main:
        tfr_main.pick_channels([elec])

    #create ANOVA table "faire evoluer pour plusieurs elecs a la fois
    tableau_main = np.zeros(shape=(n_sujets,fmax-fmin+1))
    for i in range(n_sujets):#sujets
        print("sujet"+str(i))
        #ecraser forme electrodes
        print(liste_tfr_main[i].data.shape)
        liste_tfr_main[i].data = np.mean(liste_tfr_main[i].data,axis=0)
        print(liste_tfr_main[i].data.shape)
        #pool time
        powerFreq_main = np.median(liste_tfr_main[i].data,axis=1)
        print(powerFreq_main)
        for j in range(fmax-fmin+1):#freq
            tableau_main[i][j] = powerFreq_main[j]
        print(tableau_main)
    return tableau_main
    
obj_channels=["Fp1","Fp2","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8",
"CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","O1","Oz","O2"]
liste_MIalone = []

#reorder channels

for elec in obj_channels:
    print("ELEC  "+elec)
    liste_power_sujets = load_tfr_data_windows(liste_rawPath_rawMIalone,"",True)
    #liste_power_sujets = copy_tfrs(liste_power_sujets)
    tableau_MIalone = data_freq_tTest_perm(elec,3,84,2.5,26.5,liste_power_sujets)
    liste_MIalone.append(tableau_MIalone)

    
    
  
#now get the p values
def get_pvalue_allElec_allFreq(liste_condition,npermut):
    n_sujets = liste_condition[0].shape[0]
    print(n_sujets)
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

import pandas as pd

readable_pValue_table_MIalone=  get_pvalue_allElec_allFreq(liste_MIalone,20000)  


header_row =  ["Channels\\freq"] + list(np.arange(3,85,1))  # Adding an empty cell for the top-left corner
header_col = ["Channels\\freq"] + obj_channels


# Creating a DataFrame from the data
df_pval_MIalone = pd.DataFrame(readable_pValue_table_MIalone, index=header_col[1:], columns=header_row[1:])


path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/rdom_scriptsData/allElecFreq_VSZero/MIalone_elecFixed/"

df_pval_MIalone.to_csv(path+"p_mialone.csv")


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

d_mialone = get_dcohen_allElec_allFreq(liste_MIalone)

# Creating a DataFrame from the data
d_mialone = pd.DataFrame(d_mialone, index=header_col[1:], columns=header_row[1:])

d_mialone.to_csv(path+"dcohen_mialone.csv")

    

#DISPLAY
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/rdom_scriptsData/allElecFreq_VSZero/MIalone_elecFixed/"

p_mialone = pd.read_csv(path+"p_mialone.csv").iloc[:, 1:]
p_mialone = p_mialone.to_numpy()

mialone = pd.read_csv(path+"dcohen_mialone.csv").iloc[:, 1:]
mialone = mialone.to_numpy()


import imagesc
imagesc.plot(mialone,cmap="Blues")


pvalue = 0.05
masked_mi = np.ma.masked_where((p_mialone > pvalue) , mialone)

imagesc.plot(-masked_mi,cmap="Blues")


import matplotlib.pyplot as plt
elec_leg = pd.read_csv(path+"dcohen_mialone.csv").iloc[:, 0]


vmin = 0.9
vmax = 2.1


# Create the plot
fig, ax = plt.subplots(figsize=(10, 7))

# Plot the image
img = ax.imshow(-masked_mi, extent=[0, 1, 0, 1], cmap="Blues", aspect='auto', interpolation='none', vmin=vmin, vmax=vmax, label="pendulum")
ax.text(0.12, 1.02, 'MI alone')

# Add colorbar
cbar = fig.colorbar(img, location='right')

# Set ticks and labels
elecs = elec_leg
freq_leg = np.arange(3, 84, 4)
freq_leg_str = [str(f) for f in freq_leg]
ax.set_xticks(np.linspace(0, 1, 21))
ax.set_xticklabels(freq_leg_str)
ax.set_yticks(np.linspace(1 / (len(elecs) * 2.5), 1 - 1 / (len(elecs) * 2.5), len(elecs)))
ax.set_yticklabels(elecs.iloc[::-1])

# Add vertical lines
x8Hz = 0.061
x30Hz = 0.34
col = "black"
ls = "--"
lw = 0.7
for x in [x8Hz, x30Hz]:
    ax.axvline(x=x, color=col, ls=ls, lw=lw)

# Add horizontal lines
for elecPos in [0.107, 0.286, 0.428, 0.608, 0.75, 0.9293]:
    ax.axhline(y=elecPos, color="dimgray", lw=0.25)

plt.show()
