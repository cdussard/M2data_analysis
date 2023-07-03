# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 12:31:00 2023

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
listeTfr_cond_exclude_bad = load_tfr_data_windows(liste_rawPathMain[0:1],"alltrials",True)

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


# run_dispo_main = []
# for jete in jetes_main:
#     run1 = all(element in jete for element in [1,2,3,4,5])
#     run2 = all(element in jete for element in [6,7,8,9,10])
#     run_dispo_main.append([run1,run2])
    
# run_dispo_pendule = []
# for jete in jetes_pendule:
#     run1 = all(element in jete for element in [1,2,3,4,5])
#     run2 = all(element in jete for element in [6,7,8,9,10])
#     run_dispo_pendule.append([run1,run2])

# run_dispo_mainIll = []
# for jete in jetes_mainIllusion:
#     run1 = all(element in jete for element in [1,2,3,4,5])
#     run2 = all(element in jete for element in [6,7,8,9,10])
#     run_dispo_mainIll.append([run1,run2])
    
#modify so you get one value per run


# essaisJetes_cond = jetes_main

def getDispo_cond(essaisJetes_cond):
    liste_tfr_allsujets_run1 = []
    liste_tfr_allsujets_run2 = []
    # run_dispo_cond = []
    # for jete in essaisJetes_cond:
    #     run1 = all(element in jete for element in [1,2,3,4,5])
    #     run2 = all(element in jete for element in [6,7,8,9,10])
    #     run_dispo_cond.append([run1,run2])
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
    
    
#compte_jetes = [0 for i in range(23)]

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

def cropTime(liste_tfr):
    liste_cropped = []
    for (tfr,i) in zip(liste_tfr,range(23)):
        print("sujet "+str(i))
        if not isinstance(tfr, list):
            tfr.crop(tmin=2.5,tmax=26.5)
            #tfr.pick_channels(["C3"])
            vals = np.mean(tfr.data,axis=2)
        else:
            vals = []
        liste_cropped.append(vals)
    return liste_cropped

# all_listes_timeeleccrop_run1 = cropTime (all_listes_run1)
# all_listes_timeeleccrop_run2 = cropTime (all_listes_run2)


def get_data_cond(jetes_cond,liste_path_cond,save,cond):
    #get data
    listeTfr_cond_exclude_bad = load_tfr_data_windows(liste_path_cond,"alltrials",True)
    #get dispo
    liste_tfr_allsujets_run1,liste_tfr_allsujets_run2 = getDispo_cond(jetes_cond)
    #extract trials per run
    all_listes_run1,all_listes_run2 = gd_average_allEssais_V3(listeTfr_cond_exclude_bad,True,True,liste_tfr_allsujets_run1,liste_tfr_allsujets_run2)
    #cropTime
    all_listes_timeeleccrop_run1 = cropTime (all_listes_run1)
    all_listes_timeeleccrop_run2 = cropTime (all_listes_run2)
    for (datarun1,datarun2,i) in zip(all_listes_timeeleccrop_run1,all_listes_timeeleccrop_run2,range(23)):
        print("saving subj"+str(i))
        path_sujet = liste_path_cond[i]#attention ne marche que si on a les epochs dans l'ordre
        charac_split = "\\"
        path_raccourci = str(path_sujet)[0:len(str(path_sujet))-4]
        path_raccourci_split = path_raccourci.split(charac_split)
        directory = "../MATLAB_DATA/" + path_raccourci_split[0] + "/"
        io.savemat(directory+ path_raccourci_split[0] +cond+ "run1"+".mat", {'data': datarun1 })
        io.savemat(directory+ path_raccourci_split[0] +cond+ "run2"+".mat", {'data': datarun2 })
        
    return all_listes_timeeleccrop_run1,all_listes_timeeleccrop_run2


all_listes_timeeleccrop_run1_main,all_listes_timeeleccrop_run2_main = get_data_cond(jetes_main,liste_rawPathMain,True,"main")
all_listes_timeeleccrop_run1_pendule,all_listes_timeeleccrop_run2_pendule = get_data_cond(jetes_pendule,liste_rawPathPendule,True,"pendule")
all_listes_timeeleccrop_run1_mainIllusion,all_listes_timeeleccrop_run2_mainIllusion = get_data_cond(jetes_mainIllusion,liste_rawPathMainIllusion,True,"mainIllusion")


#construct the dictionnary
freqs = np.arange(3, 84, 1)
all_data = ["pendule", "main", "mainIllusion"]
channels = ["Fp1", "Fp2", "F7", "F3", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
            "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2","Fz"]

n_elec = len(channels)
n_freq = len(freqs)

num_iterations = len(listeNumSujetsFinale) * len(all_data) * n_elec * n_freq*2
dict_total = np.empty((num_iterations, 6), dtype=object)
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

dict_total,index = add_to_dict("run1","main",all_listes_timeeleccrop_run1_main,dict_total,0)
dict_total,index  = add_to_dict("run2","main",all_listes_timeeleccrop_run2_main,dict_total,index )
dict_total,index  = add_to_dict("run1","pendule",all_listes_timeeleccrop_run1_pendule,dict_total,index )
dict_total,index  = add_to_dict("run2","pendule",all_listes_timeeleccrop_run2_pendule,dict_total,index )
dict_total,index  = add_to_dict("run1","mainIllusion",all_listes_timeeleccrop_run1_mainIllusion,dict_total,index )
dict_total,index  = add_to_dict("run2","mainIllusion",all_listes_timeeleccrop_run2_mainIllusion,dict_total,index )
dict_total_df = pd.DataFrame(dict_total[:index], columns=["num_sujet", "FB", "elec", "freq", "ERD_value","run"])

path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/fig_brian/correlERD_agency_colorgrid/V3_2runs/"

dict_total_df.to_csv(path+"optimized_dataCorrel.csv")      

# path = "../../../../MATLAB_DATA/"
# for suj,num_suj in zip(listeNumSujetsFinale,num_sujets):
#     for FB in all_data:
#         path_sujet_fb1 = path + suj + "/" + suj + FB + "run1.mat"
#         print(path_sujet_fb1)
#         data1 = loadmat(path_sujet_fb1)["data"]
#         path_sujet_fb2 = path + suj + "/" + suj + FB + "run2.mat"
#         print(path_sujet_fb2)
#         data2 = loadmat(path_sujet_fb2)["data"]
#         for elec_i in range(n_elec):
#             for freq_i in range(n_freq):
#                 value = data1[elec_i, freq_i]
#                 dict_total[index] = [num_suj, FB, channels[elec_i], freq_i + 3, value,"run1"]
#                 index += 1
#         for elec_i in range(n_elec):
#             for freq_i in range(n_freq):
#                 value = data2[elec_i, freq_i]
#                 dict_total[index] = [num_suj, FB, channels[elec_i], freq_i + 3, value,"run2"]
#                 index += 1

# dict_total = pd.DataFrame(dict_total[:index], columns=["num_sujet", "FB", "elec", "freq", "ERD_value","run"])


# num_sujets = allSujetsDispo
# path = "../../../../MATLAB_DATA/"
# for suj,num_suj in zip(listeNumSujetsFinale,num_sujets):
#     for FB in all_data:
        
#         data1 = #loadmat(path_sujet_fb1)["data"]

#         data2 = #loadmat(path_sujet_fb2)["data"]
#         for elec_i in range(n_elec):
#             for freq_i in range(n_freq):
#                 value = data1[elec_i, freq_i]
#                 dict_total[index] = [num_suj, FB, channels[elec_i], freq_i + 3, value,"run1"]
#                 index += 1
#         for elec_i in range(n_elec):
#             for freq_i in range(n_freq):
#                 value = data2[elec_i, freq_i]
#                 dict_total[index] = [num_suj, FB, channels[elec_i], freq_i + 3, value,"run2"]
#                 index += 1

dict_total = pd.DataFrame(dict_total[:index], columns=["num_sujet", "FB", "elec", "freq", "ERD_value","run"])