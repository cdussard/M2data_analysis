# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:52:07 2024

@author: claire.dussard
"""

import mne

from functions.load_savedData import *
from handleData_subject import createSujetsData
from functions.load_savedData import *
import numpy as np
import os
import pandas as pd
from mne.time_frequency.tfr import combine_tfr
import scipy
from scipy import io
from scipy.io import loadmat

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()

#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

cond = "main"
listeTfr_cond_exclude_bad = load_tfr_data_windows(liste_rawPathMain[0:2],"alltrials",True)

#recenser les runs dispo

jetes_main = [
    [],[],[],[3],[2,4,5,6,7],[7],[],[6],[9,10],[8],[6],
    [1,6,8],[1,10],[9,10],[6,7,8,9,10],[3,6],[3,6,7],[4,10],[],[1,6],[],[9],[]
    ]

jetes_pendule = [
    [],[],[],[5],[1,7,10],[],[],[3,5,8,10],[],[5,10],[],
    [5,6],[4],[6,9],[],[9],[3,8,9],[],[],[1,6],[6],[3,9],[6,8]
    ]

jetes_mainIllusion = [
    [6],[1,3,6],[1,2],[],[5,6,8,9,10],[],[],[1,6,7,8],[6,7,8,9,10],[4,10],[1],
    [],[1,8,10],[10],[6,9],[9],[4,8,9],[4,8],[],[1,6],[],[1],[]
    ]


essaisJetes = jetes_main

index_essais = [[x - 1 for x in sorted(set(range(1, 11)) - set(jetes))] for jetes in jetes_main]

    
#compte_jetes = [0 for i in range(23)]

def gd_average_allEssais_V3(liste_tfr_cond,index_essais,doBaseline):
    dureePreBaseline = 3 #3
    dureePreBaseline = - dureePreBaseline
    dureeBaseline = 2.0 #2.0
    valeurPostBaseline = dureePreBaseline + dureeBaseline
    baseline = (dureePreBaseline, valeurPostBaseline)
    all_listes_trials = []
    for i in range(len(liste_tfr_cond)):
        print("num sujet" + str(i))
        print(index_essais[i])
        liste_tfr_suj = liste_tfr_cond[i]
        essais_run1 = [liste_tfr_suj[i] for i in range(len(index_essais[i]))]
        if doBaseline:
            for essai in essais_run1:
                    essai.apply_baseline(baseline=baseline, mode='zscore')
        all_listes_trials.append(essais_run1)
    return all_listes_trials
        
all_listes = gd_average_allEssais_V3(listeTfr_cond_exclude_bad,index_essais)

def cropTime(liste_tfr):
    liste_cropped = []
    for (trials_tfr,i) in zip(liste_tfr,range(23)):
        print("sujet "+str(i))
        suj_table = []
        for trial in trials_tfr:
            trial.crop(tmin=2.5,tmax=26.5)
            vals = np.mean(trial.data,axis=0)
            vals = np.mean(vals,axis=2)#MOYENNER LE TEMPS
            print(vals.shape)
            suj_table.append(vals)
        liste_cropped.append(suj_table)
    return liste_cropped

all_listes_timeeleccrop = cropTime (all_listes)



def cropTimeV2(liste_tfr,getCycles):
    liste_cropped = []
    for (trials_tfr,i) in zip(liste_tfr,range(23)):
        print("sujet "+str(i))
        suj_table = []
        for trial in trials_tfr:
            trial.crop(tmin=2.5,tmax=26.5)
            vals = np.mean(trial.data,axis=0)
            #print(vals.shape)
            if getCycles:
                vals = computeMovingAverage2D(vals, 24)  # 24 for cropped data
            else:
                vals = np.mean(vals,axis=2)#MOYENNER LE TEMPS
            print(vals.shape)
            suj_table.append(vals)
        liste_cropped.append(suj_table)
    return liste_cropped

all_listes_timeeleccrop = cropTimeV2(all_listes,True)


def computeMovingAverage2D(C3values, nvalues):
    """
    Compute the moving average for a 3D array (28, 82, time),
    dropping None values from the result.
    """
    offsets = [0, 63, 125, 188]
    valid_results = []  # List to collect valid moving averages

    compteur_moyenne = 1  # Counter to track skipping logic
    for i in range(1, nvalues):
        if compteur_moyenne == 5:
            compteur_moyenne += 1
            continue
        elif compteur_moyenne == 6:
            compteur_moyenne = 1
            continue

        # Compute the start and end indices for the four sub-windows
        ranges = [(offset + 250 * i, offset + 250 * (i + 1)) for offset in offsets]
        
        # Slice and compute mean across the last axis for all (28, 82)
        means = [C3values[:, :, start:end].mean(axis=2) for start, end in ranges]
        
        # Average the means
        point_moyenne = sum(means) / len(means)

        # Append the result
        valid_results.append(point_moyenne)

        compteur_moyenne += 1
    
    # Convert list of valid results to a NumPy array
    return np.stack(valid_results, axis=-1)
    

def get_data_cond(jetes_cond,liste_path_cond,save,cond,getCycles,doBaseline):
    #get data
    listeTfr_cond_exclude_bad = load_tfr_data_windows(liste_path_cond,"alltrials",True)
    #get dispo
    index_essais = [[x - 1 for x in sorted(set(range(1, 11)) - set(jetes))] for jetes in jetes_cond]
    #extract trials per run
    all_listes = gd_average_allEssais_V3(listeTfr_cond_exclude_bad,index_essais,doBaseline)
    #cropTime
    all_listes_timeeleccrop = cropTimeV2(all_listes,getCycles)
    if save:
        for (data_trials,i) in zip(all_listes_timeeleccrop,range(23)):
            print("saving subj"+str(i))
            path_sujet = liste_path_cond[i]#attention ne marche que si on a les epochs dans l'ordre
            charac_split = "\\"
            path_raccourci = str(path_sujet)[0:len(str(path_sujet))-4]
            path_raccourci_split = path_raccourci.split(charac_split)
            directory = "../MATLAB_DATA/" + path_raccourci_split[0] + "/"
            for (essai,realIndex) in zip(index_essais[i],range(len(index_essais[i]))):
                 io.savemat(directory+ path_raccourci_split[0] +cond+ "essai"+str(essai)+".mat", {'data': data_trials[realIndex] })
            
    return all_listes_timeeleccrop


all_listes_timeeleccrop_main = get_data_cond(jetes_main,liste_rawPathMain,True,"main")
all_listes_timeeleccrop_pendule = get_data_cond(jetes_pendule,liste_rawPathPendule,True,"pendule")
all_listes_timeeleccrop_mainIllusion = get_data_cond(jetes_mainIllusion,liste_rawPathMainIllusion,True,"mainIllusion")


all_listes_timeeleccrop_main = get_data_cond(jetes_main,liste_rawPathMain,False,"main",True,True)#cycles data
all_listes_timeeleccrop_pendule = get_data_cond(jetes_pendule,liste_rawPathPendule,False,"pendule",True,True)#cycles data
all_listes_timeeleccrop_mainIllusion = get_data_cond(jetes_mainIllusion,liste_rawPathMainIllusion,False,"mainIllusion",True,True)#cycles data

all_listes_timeeleccrop_main_noBL = get_data_cond(jetes_main,liste_rawPathMain,False,"main",True,False)#cycles data
all_listes_timeeleccrop_pendule_noBL = get_data_cond(jetes_pendule,liste_rawPathPendule,False,"pendule",True,False)#cycles data
all_listes_timeeleccrop_mainIllusion_noBL = get_data_cond(jetes_mainIllusion,liste_rawPathMainIllusion,False,"mainIllusion",True,False)#cycles data

#construct the dictionnary
freqs = np.arange(3, 84, 1)
all_data = ["pendule", "main", "mainIllusion"]
channels = ["Fp1", "Fp2", "F7", "F3", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
            "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2","Fz"]

n_elec = len(channels)
n_freq = len(freqs)
n_cycles = 16

num_iterations = len(listeNumSujetsFinale) * len(all_data) * n_elec * n_freq*2*10*n_cycles
dict_total = np.empty((num_iterations, 7), dtype=object)

num_sujets = allSujetsDispo

def add_to_dict(FB,data,dico,jetes_cond,index,getCycles):
    index_essais = [[x - 1 for x in sorted(set(range(1, 11)) - set(jetes))] for jetes in jetes_cond]
    for suj,num_suj,i in zip(listeNumSujetsFinale,num_sujets,range(23)):
        print(i)
        index_suj = index_essais[i]
        data_suj = data[i]
        print(len(data_suj))
        if not getCycles:
            for (trial,trueIndex_trial) in zip(data_suj,index_suj):
                for elec_i in range(n_elec):
                        for freq_i in range(n_freq):
                            if not isinstance(trial, list):
                                value = trial[elec_i, freq_i]#idealement checker les bons index ch & freq
                            else:
                                value = "NA"
                            dico[index] = [num_suj, FB, channels[elec_i], freq_i + 3, value,trueIndex_trial]
                            index += 1
        elif getCycles:
            for (trial,trueIndex_trial) in zip(data_suj,index_suj):
                for elec_i in range(n_elec):
                        for freq_i in range(n_freq):
                            for cycle in range(n_cycles):
                                if not isinstance(trial, list):
                                    value = trial[elec_i, freq_i,cycle]#idealement checker les bons index ch & freq
                                else:
                                    value = "NA"
                                dico[index] = [num_suj, FB, channels[elec_i], freq_i + 3, value,trueIndex_trial,cycle]
                                index += 1
                
    return dico,index

index = 0
dict_total,index = add_to_dict("main",all_listes_timeeleccrop_main,dict_total,jetes_main,index)
dict_total,index  = add_to_dict("pendule",all_listes_timeeleccrop_pendule,dict_total,jetes_pendule,index)
dict_total,index  = add_to_dict("mainIllusion",all_listes_timeeleccrop_mainIllusion,dict_total,jetes_mainIllusion,index)
dict_total_df = pd.DataFrame(dict_total[:index], columns=["num_sujet", "FB", "elec", "freq", "ERD_value","trial"])
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/"

dict_total_df.to_csv(path+"optimized_dataCorrel_allTrials.csv")      


index = 0
dict_total,index = add_to_dict("main",all_listes_timeeleccrop_main,dict_total,jetes_main,index,True)
dict_total,index  = add_to_dict("pendule",all_listes_timeeleccrop_pendule,dict_total,jetes_pendule,index,True)
dict_total,index  = add_to_dict("mainIllusion",all_listes_timeeleccrop_mainIllusion,dict_total,jetes_mainIllusion,index,True)


dict_total_df = pd.DataFrame(dict_total[:index], columns=["num_sujet", "FB", "elec", "freq", "ERD_value","trial","cycle"])
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/"

dict_total_df.to_csv(path+"optimized_dataCorrel_allTrials_allCycles.csv")     


index = 0
dict_total,index = add_to_dict("main",all_listes_timeeleccrop_main_noBL,dict_total,jetes_main,index,True)
dict_total,index  = add_to_dict("pendule",all_listes_timeeleccrop_pendule_noBL,dict_total,jetes_pendule,index,True)
dict_total,index  = add_to_dict("mainIllusion",all_listes_timeeleccrop_mainIllusion_noBL,dict_total,jetes_mainIllusion,index,True)


dict_total_df = pd.DataFrame(dict_total[:index], columns=["num_sujet", "FB", "elec", "freq", "ERD_value","trial","cycle"])
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/"

dict_total_df.to_csv(path+"optimized_dataCorrel_allTrials_allCycles_noBL.csv")  


#==========================DISPLAY FIGURE !!!!!!!!!!!==============
import pandas as pd
import numpy as np

# Path to the directory containing the CSV files
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/BCImeeting2025/graz/"

# Read the CSV files
df_global = pd.read_csv(path + "estimate_predPerfbyERD_trials.csv", header=None, delimiter=",").iloc[1:, 1:].values
df_global_pval = pd.read_csv(path + "pval_predPerfbyERD_trials.csv", header=None, delimiter=",").iloc[1:, 1:].values

df_global = pd.read_csv(path + "estimate_predPerfbyERD_trials_cycles.csv", header=None, delimiter=",").iloc[1:, 1:].values
df_global_pval = pd.read_csv(path + "pval_predPerfbyERD_trials_cycles.csv", header=None, delimiter=",").iloc[1:, 1:].values

df_global = df_global.astype(float)
df_global_pval = df_global_pval.astype(float)


#subset from fmin to fmax
fmin = 3
fmax = 40
len_to_keep = fmax - fmin
df_subset = df_global[:,0:len_to_keep+1]
df_pval_subset = df_global_pval[:,0:len_to_keep+1]
print(df_pval_subset.min())
vmin = -0.022
vmax = -vmin
cmap = "RdBu_r"


corrected_pval = mne.stats.fdr_correction(df_pval_subset)[1]
print(corrected_pval.min())
#FDR CORR
pvalue = 0.05
masked_global = np.ma.masked_where((corrected_pval > pvalue) , df_subset)
masked_global.data
df_subset[corrected_pval < pvalue]


import matplotlib.pyplot as plt
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/old_papers/rdom_scriptsData/allElecFreq_VSZero/versionJuin2023_elecFixed/"
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

av_tfr =  mne.time_frequency.read_tfrs("AV_TFR/all_sujets/MIalone_logratio-tfr.h5")[0]

#QUE POUR 32
ch_names = ["Fp1", "Fp2", "F7" , "F3", "Fz", "F4",  "F8"  ,"FC5", "FC1", "FC2","FC6","T7","C3","Cz","C4","T8","CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","O1","Oz","O2"]
av = av_tfr.pick_channels(ch_names)#QUE POUR 32
av = av.reorder_channels(ch_names)
info = av.info

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

my_cmap = discrete_cmap(13, cmap)
vmax=0.017
vmin=-vmax
plot_topomapV2(3,7,df_global,vmin,vmax,my_cmap)
plot_topomapV2(8,12,df_global,vmin,vmax,my_cmap)
plot_topomapV2(12,15,df_global,vmin,vmax,my_cmap)
plot_topomapV2(13,20,df_global,vmin,vmax,my_cmap)
plot_topomapV2(20,30,df_global,vmin,vmax,my_cmap)
plot_topomapV2(8,30,df_global,vmin,vmax,my_cmap)


dict_total = pd.DataFrame(dict_total[:index], columns=["num_sujet", "FB", "elec", "freq", "ERD_value","run"])