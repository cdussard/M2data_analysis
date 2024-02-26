# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 19:36:53 2024

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
from functions_graz import *

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()

#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

cond = "main"
#listeTfr_cond_exclude_bad = load_tfr_data_windows(liste_rawPathMainIllusion[0:1],"alltrials",True)[0]




#============== essayer d'exclure les parties avec vibration ========

#recenser les runs dispo

jetes_mainIllusion = [
    [6],[1,3,6],[1,2],[],[5,6,8,9,10],[],[],[1,6,7,8],[6,7,8,9,10],[4,10],[1],
    [],[1,8,10],[10],[6,9],[9],[4,8,9],[4,8],[],[1,6],[],[1],[]
    ]



def getDispo_cond(essaisJetes_cond):
    liste_tfr_allsujets_run1 = []
    liste_tfr_allsujets_run2 = []

    compte_jetes = [0 for i in range(23)]
    for j in range(23):#n_sujets
        print("sujet "+str(j))
        essais_jetes_suj = essaisJetes_cond[j]
        print("essais jetes")
        print(essais_jetes_suj)
        #run_dispo = run_dispo_cond[j]
        liste_tfr_sujet_run1 = []
        liste_tfr_sujet_run2 = []
        for i in range(10):
            if i+1 not in essais_jetes_suj:#CENSE AVOIR DEUX PARTIES ??
                print("dispo")
                index = i-compte_jetes[j]
                if i+1<6:
                    print("run 1 ")
                    liste_tfr_sujet_run1.append(index)
                else:
                    print("run 2 ")
                    liste_tfr_sujet_run2.append(index) 
            else:
                compte_jetes[j] += 1 
                print("essai jete")
                index = i-compte_jetes[j]
     
            print(liste_tfr_sujet_run1)
            print(liste_tfr_sujet_run2)
        liste_tfr_allsujets_run1.append(liste_tfr_sujet_run1)
        liste_tfr_allsujets_run2.append(liste_tfr_sujet_run2)
    return liste_tfr_allsujets_run1,liste_tfr_allsujets_run2
    

def gd_average_allEssais_V3(liste_tfr_cond,get_all_suj,baseline,index_essais_run1,index_essais_run2):
    dureePreBaseline = 3 #3
    dureePreBaseline = - dureePreBaseline
    dureeBaseline = 2.0 #2.0
    valeurPostBaseline = dureePreBaseline + dureeBaseline
    baseline = (dureePreBaseline, valeurPostBaseline)
    all_listes_run1 = []
    all_listes_run2 = []
    for i in range(len(liste_tfr_cond)):
        print("num sujet" + str(i))
        liste_tfr_suj = liste_tfr_cond[i]
        indexes_run1 = index_essais_run1[i]
        print('run 1')
        essais_run1 = [liste_tfr_suj[i] for i in indexes_run1]
        essais_run1 = [item.average() for item in essais_run1]
        if len(essais_run1)>1:
            av_essais_run1 = combine_tfr(essais_run1,'equal')
            bl_essais_run1 = av_essais_run1.apply_baseline(baseline=baseline, mode='logratio')
        else:
            bl_essais_run1 = []
        all_listes_run1.append(bl_essais_run1)
        print('run 2')
        indexes_run2 = index_essais_run2[i]
        essais_run2 = [liste_tfr_suj[i] for i in indexes_run2]
        essais_run2 = [item.average() for item in essais_run2]
        if len(essais_run2)>1:
            av_essais_run2 = combine_tfr(essais_run2,'equal')
            bl_essais_run2 = av_essais_run2.apply_baseline(baseline=baseline, mode='logratio')
        else:
            bl_essais_run2=[]
        all_listes_run2.append(bl_essais_run2)
    return all_listes_run1,all_listes_run2
        
# all_listes_run1,all_listes_run2 = gd_average_allEssais_V3(listeTfr_cond_exclude_bad,True,True,liste_tfr_allsujets_run1,liste_tfr_allsujets_run2)

# def cropTime(liste_tfr):
#     liste_cropped = []
#     for (tfr,i) in zip(liste_tfr,range(23)):
#         print("sujet "+str(i))
#         if not isinstance(tfr, list):
#             tfr.crop(tmin=2.5,tmax=26.5)
#             #tfr.pick_channels(["C3"])
#             vals = np.mean(tfr.data,axis=2)
#         else:
#             vals = []
#         liste_cropped.append(vals)
#     return liste_cropped


def cropTimeVib(liste_tfr):
    liste_cropped = []
    t_Startvib_nf = [6.2,12.4,18.6,24.8]
    durVib = 2 #•1.3 sur la derniere 
    fEch = 250
    intDurVib = durVib*fEch
    intVibss = [int((val*fEch)-2.5*fEch) for val in t_Startvib_nf]
    endVibs = [startvib + intDurVib for startvib in intVibss]
    segments_to_exclude = []
    for i in range(len(intVibss)):
        couple = (intVibss[i],endVibs[i])
        segments_to_exclude.append(couple)
        
    for (tfr,i) in zip(liste_tfr,range(23)):
        print("sujet "+str(i))
        if not isinstance(tfr, list):
            ls = tfr.crop(tmin=2.5,tmax=26.5)
            print(ls._data.shape)
            tfr_data = ls._data
            data = tfr_data
            # Create a mask to identify the indices to exclude
            mask = np.zeros(data.shape[2], dtype=bool)
            for segment in segments_to_exclude:
                start, end = segment
                mask[start:end+1] = True
            
            # Invert the mask to select the values outside of the segments
            inverted_mask = ~mask
            
            # Use the mask to select the values outside of the segments
            values_outside_segments = data[:, :, inverted_mask]
            vals = np.mean(values_outside_segments,axis=2) 
        else:
            vals = []
        liste_cropped.append(vals)
    return liste_cropped

# all_listes_timeeleccrop_run1 = cropTimeVib (all_listes_run1)
# all_listes_timeeleccrop_run2 = cropTimeVib (all_listes_run2)


def get_data_cond(jetes_cond,liste_path_cond,save,cond):
    #get data
    listeTfr_cond_exclude_bad = load_tfr_data_windows(liste_path_cond,"alltrials",True)
    #get dispo
    liste_tfr_allsujets_run1,liste_tfr_allsujets_run2 = getDispo_cond(jetes_cond)
    #extract trials per run
    all_listes_run1,all_listes_run2 = gd_average_allEssais_V3(listeTfr_cond_exclude_bad,True,True,liste_tfr_allsujets_run1,liste_tfr_allsujets_run2)
    #cropTime
    all_listes_timeeleccrop_run1 = cropTimeVib (all_listes_run1)
    all_listes_timeeleccrop_run2 = cropTimeVib (all_listes_run2)
    for (datarun1,datarun2,i) in zip(all_listes_timeeleccrop_run1,all_listes_timeeleccrop_run2,range(23)):
        print("saving subj"+str(i))
        path_sujet = liste_path_cond[i]#attention ne marche que si on a les epochs dans l'ordre
        charac_split = "\\"
        path_raccourci = str(path_sujet)[0:len(str(path_sujet))-4]
        path_raccourci_split = path_raccourci.split(charac_split)
        directory = "../MATLAB_DATA/" + path_raccourci_split[0] + "/"
        io.savemat(directory+ path_raccourci_split[0] +cond+ "run1_noVib"+".mat", {'data': datarun1 })
        io.savemat(directory+ path_raccourci_split[0] +cond+ "run2_noVib"+".mat", {'data': datarun2 })
        
    return all_listes_timeeleccrop_run1,all_listes_timeeleccrop_run2


all_listes_timeeleccrop_run1_mainIllusion,all_listes_timeeleccrop_run2_mainIllusion = get_data_cond(jetes_mainIllusion,liste_rawPathMainIllusion,True,"mainIllusion")


#construct the dictionnary
freqs = np.arange(3, 84, 1)
all_data = ["mainIllusion"]
channels = ["Fp1", "Fp2", "F7", "F3", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
            "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2","Fz"]

n_elec = len(channels)
n_freq = len(freqs)

num_iterations = len(listeNumSujetsFinale) * len(all_data) * n_elec * n_freq*2
dict_total_df_noVib = np.empty((num_iterations, 6), dtype=object)
index = 0

num_sujets = allSujetsDispo

def add_to_dict(run,FB,data,dico,index):
    for suj,num_suj,i in zip(listeNumSujetsFinale,num_sujets,range(23)):
        print(i)
        data_suj = data[i]
        print(data_suj)
        for elec_i in range(n_elec):
                for freq_i in range(n_freq):
                    print(elec_i)
                    print(freq_i)
                    if not isinstance(data_suj, list):
                        value = data_suj[elec_i, freq_i]
                    else:
                        value = "NA"
                    dico[index] = [num_suj, FB, channels[elec_i], freq_i + 3, value,run]
                    index += 1
                
    return dico,index


dict_total_df_noVib,index  = add_to_dict("run1","mainIllusion",all_listes_timeeleccrop_run1_mainIllusion,dict_total_df_noVib,index )
dict_total_df_noVib,index  = add_to_dict("run2","mainIllusion",all_listes_timeeleccrop_run2_mainIllusion,dict_total_df_noVib,index )
dict_total_df_noVib = pd.DataFrame(dict_total_df_noVib[:index], columns=["num_sujet", "FB", "elec", "freq", "ERD_value","run"])

path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/fig_brian/correlERD_agency_colorgrid/V3_2runs/"

dict_total_df_noVib.to_csv(path+"optimized_dataCorrel_noVib.csv")      



#============== now plot the fucking plots ===============


import pandas as pd
import mne
import numpy as np
import imagesc

# Path to the directory containing the CSV files
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/Article_physio/analyses/reanalysisPerf/"


# Read the CSV files
df_global = pd.read_csv(path + "df_global_estimateFBseul_v3_2runs_NFperf_sujFactor_noVib.csv", header=None, delimiter=",").iloc[1:, 1:].values
df_global_pval = pd.read_csv(path + "df_global_pvalFBseul_v3_2runs_NFperf_anova_sujFactor_noVib.csv", header=None, delimiter=",").iloc[1:, 1:].values


df_global = df_global.astype(float)
df_global_pval = df_global_pval.astype(float)

freqs = np.arange(3,84,1)
obj_channels=["Fp1","Fp2","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8",
"CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","O1","Oz","O2"]
info = mne.create_info(obj_channels,250,"eeg")
info.set_montage(mne.channels.make_standard_montage('easycap-M1'))


#subset from fmin to fmax
fmin = 3
fmax = 40
len_to_keep = fmax - fmin
df_subset = df_global[:,0:len_to_keep+1]
df_pval_subset = df_global_pval[:,0:len_to_keep+1]
print(df_pval_subset.min())
vmin = -0.35#-0.35
vmax = -vmin
cmap = "RdBu_r"


#FDR correction

corrected_pval = mne.stats.fdr_correction(df_pval_subset)[1]
print(corrected_pval.min())

pvalue = 0.05

masked_global = np.ma.masked_where((corrected_pval > pvalue) , df_subset)
masked_global.data
df_subset[corrected_pval < pvalue]
plot_fig_elec(masked_global,vmin,vmax,cmap)

vmax = 0.4
vmin = -vmax
fig, axs = plt.subplots(1,5, sharey=True,sharex=True, figsize=(14, 7),constrained_layout=True)
plot_topomapV3(3,7,df_subset,vmin,vmax,my_cmap,axs,0)
plot_topomapV3(8,12,df_subset,vmin,vmax,my_cmap,axs,1)
plot_topomapV3(12,15,df_subset,vmin,vmax,my_cmap,axs,2)
plot_topomapV3(13,20,df_subset,vmin,vmax,my_cmap,axs,3)
im = plot_topomapV3(21,30,df_subset,vmin,vmax,my_cmap,axs,4)
fig.colorbar(im, location = 'right', shrink=0.5)

plot_topomapV2(8,30,df_subset,-0.35,0.35,my_cmap)










