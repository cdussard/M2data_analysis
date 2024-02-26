# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 19:05:09 2024

@author: claire.dussard
"""
#on commence par extraire le raw

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
from joblib import Parallel, delayed

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()

#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)



#on construit les temps sur lesquels on va filtrer
fEch = 250
tempsCycles = [0.2]
for i in range(16):
    tempsCycles.append(tempsCycles[0]+(i+1)*1.55)
    
lenCycle = 1.5


EchsCycles = [int(temp * fEch) for temp in tempsCycles]
lenEchsCycles = int(lenCycle *fEch)
listeTfr_cond_withBad= load_tfr_data_windows(liste_rawPathMain[0:1],"alltrials_includingbad",True)
listeTfr_cond_withBad[0].drop_channels(["TP10","TP9","FT9","FT10"])

elec_names = listeTfr_cond_withBad[0].info.ch_names




data_main = [] 

for suj, suj_name in enumerate(allSujetsDispo):
    tfrs_suj = load_one_tfr_windows(liste_rawPathMain[suj], "alltrials_includingbad")
    tfrs_suj.drop_channels(["TP10","TP9","FT9","FT10"])
    print("suj "+str(suj))
    for trial in range(tfrs_suj._data.shape[0]):
        print("trial "+str(trial))
        tfr_essai = tfrs_suj[trial]
        for cycle in range(16):#crop le bon moment sans alterer l'objet
            cycle_data = []
            for elec, elec_name in enumerate(elec_names):
                print("elec "+str(elec_names[elec]))
                for freq in range(82):
                    arrElecFreq = tfr_essai._data[0, elec, freq]
                    val = arrElecFreq[EchsCycles[cycle]:EchsCycles[cycle] + lenEchsCycles].mean()
                    cycle_data.append((suj, trial + 1, cycle + 1, "main", freq + 3, elec_name, val))
            data_main.extend(cycle_data)
 

df_main = pd.DataFrame(data_main, columns=['num_sujet', 'essai_tot', 'cycle', 'FB', 'freq', 'elec', 'rawPower'])
df_main.to_csv("C:/Users/claire.dussard/OneDrive - ICM\Bureau/Article_physio/data/df_allFreqElec_main2.csv")

data_pendule = [] 

for suj, suj_name in enumerate(allSujetsDispo):
    tfrs_suj = load_one_tfr_windows(liste_rawPathPendule[suj], "alltrials_includingbad")
    tfrs_suj.drop_channels(["TP10","TP9","FT9","FT10"])
    print("suj "+str(suj))
    for trial in range(tfrs_suj._data.shape[0]):
        print("trial "+str(trial))
        tfr_essai = tfrs_suj[trial]
        for cycle in range(16):#crop le bon moment sans alterer l'objet
            cycle_data = []
            for elec, elec_name in enumerate(elec_names):
                print("elec "+str(elec_names[elec]))
                for freq in range(82):
                    arrElecFreq = tfr_essai._data[0, elec, freq]
                    val = arrElecFreq[EchsCycles[cycle]:EchsCycles[cycle] + lenEchsCycles].mean()
                    cycle_data.append((suj, trial + 1, cycle + 1, "pendule", freq + 3, elec_name, val))
            data_pendule.extend(cycle_data)
 

df_pend = pd.DataFrame(data_pendule, columns=['num_sujet', 'essai_tot', 'cycle', 'FB', 'freq', 'elec', 'rawPower'])
df_pend.to_csv("C:/Users/claire.dussard/OneDrive - ICM\Bureau/Article_physio/data/df_allFreqElec_pend2.csv")

data_mainvib = [] 

for suj, suj_name in enumerate(allSujetsDispo):
    tfrs_suj = load_one_tfr_windows(liste_rawPathMainIllusion[suj], "alltrials_includingbad")
    tfrs_suj.drop_channels(["TP10","TP9","FT9","FT10"])
    print("suj "+str(suj))
    for trial in range(tfrs_suj._data.shape[0]):
        print("trial "+str(trial))
        tfr_essai = tfrs_suj[trial]
        for cycle in range(16):#crop le bon moment sans alterer l'objet
            cycle_data = []
            for elec, elec_name in enumerate(elec_names):
                print("elec "+str(elec_names[elec]))
                for freq in range(82):
                    arrElecFreq = tfr_essai._data[0, elec, freq]
                    val = arrElecFreq[EchsCycles[cycle]:EchsCycles[cycle] + lenEchsCycles].mean()
                    cycle_data.append((suj, trial + 1, cycle + 1, "mainvib", freq + 3, elec_name, val))
            data_mainvib.extend(cycle_data)
 

df_mainvib = pd.DataFrame(data_mainvib, columns=['num_sujet', 'essai_tot', 'cycle', 'FB', 'freq', 'elec', 'rawPower'])

df_mainvib.to_csv("C:/Users/claire.dussard/OneDrive - ICM\Bureau/Article_physio/data/df_allFreqElec_mainvib2.csv")



#code de modeleMixteAgencyERDV2 pr s'inspirer


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
