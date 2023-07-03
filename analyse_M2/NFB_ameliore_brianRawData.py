# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 13:06:29 2023

@author: claire.dussard
"""

import os 
import seaborn as sns
import pathlib
import mne
#necessite d'avoir execute handleData_subject.py, et load_savedData avant 
import numpy as np 
# importer les fonctions definies par moi 
from handleData_subject import createSujetsData
from functions.load_savedData import *
from functions.preprocessData_eogRefait import *

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()

#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)



liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

 
#ON REFAIT TOUT PROPREMENT 
#event_id = {'Essai_mainTactile':3}
event_id_mainIllusion = {'Essai_mainIllusion':3}
event_id_pendule={'Essai_pendule':4}  
event_id_main={'Essai_main':3}  


nbSujets = 24
SujetsDejaTraites = 0
rawPath_main_sujets = liste_rawPathMain[SujetsDejaTraites:SujetsDejaTraites+nbSujets]
rawPath_pendule_sujets = liste_rawPathPendule[SujetsDejaTraites:SujetsDejaTraites+nbSujets]
rawPath_mainIllusion_sujets = liste_rawPathMainIllusion[SujetsDejaTraites:SujetsDejaTraites+nbSujets]


#on load les data apres ICA mais avant drop des bad trials
EpochDataMain_dropBad = load_data_postICA_preDropbad(rawPath_main_sujets,"",True)
EpochDataPendule_dropBad = load_data_postICA_preDropbad(rawPath_pendule_sujets,"",True)
EpochDataMainIllusion_dropBad = load_data_postICA_preDropbad(rawPath_mainIllusion_sujets,"",True)


#add virtual reference and re reference data

def full_average_ref(liste_postIca,initial_ref):
    channels = ['VEOG','HEOG','Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9','CP5','CP1','CP2','CP6','TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
    listeAverageRef = average_rereference(liste_postIca,initial_ref)
    #listeMontaged = montage_eeg(listeAverageRef)
    listeEpochs_bonOrdreChannels = change_order_channels(channels,listeAverageRef)
    return listeEpochs_bonOrdreChannels
     

initial_ref = 'Fz'
liste_epochs_averageRef_main = full_average_ref(EpochDataMain_dropBad,initial_ref)
liste_epochs_averageRef_pendule = full_average_ref(EpochDataPendule_dropBad,initial_ref)
liste_epochs_averageRef_mainIllusion = full_average_ref(EpochDataMainIllusion_dropBad,initial_ref)


#===================set montage===IMPORTANT!!!!=======================
montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
for epochs in liste_epochs_averageRef_main:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)
for epochs in liste_epochs_averageRef_pendule:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)
for epochs in liste_epochs_averageRef_mainIllusion:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)
        
        
        
liste_power_sujets = []
freqs = np.arange(3, 85, 1)
n_cycles = freqs 
i = 0
EpochData = liste_epochs_averageRef_mainIllusion
EpochData = liste_epochs_averageRef_main
EpochData = liste_epochs_averageRef_pendule


for epochs_sujet in EpochData:
    print("========================\nsujet"+str(i))
    epochData_sujet_down = epochs_sujet.resample(250., npad='auto') 
    print("downsampling...")
    power_sujet = mne.time_frequency.tfr_morlet(epochData_sujet_down,freqs=freqs,n_cycles=n_cycles,return_itc=False,average=False)#ON VEUT TOUS LES TRIALS
    print("computing power...")
    liste_power_sujets.append(power_sujet)
    i += 1
    
#================== SAVE ALL THE FUCKING DATA ======================  
save_tfr_data(liste_power_sujets,liste_rawPathMain,"alltrials_includingbad",True)#A LANCER APRES

 
save_tfr_data(liste_power_sujets,liste_rawPathMainIllusion,"alltrials_includingbad",True)


save_tfr_data(liste_power_sujets,liste_rawPathPendule,"alltrials_includingbad",True)


listeTfrAv_main = load_tfr_data_windows(liste_rawPathMain[0:2],"",True)


listeTfr_main = load_tfr_data_windows(liste_rawPathMain[0:2],"alltrials_includingbad",True)

#======= now we have the data ===============
#======== plot one topo plot in alpha and low beta and high beta per trial averaged over participants====
#============plot one topo plot same bands but for 1-3 trials and 8-10 trials for each feedback==========


#gd_average = mne.grand_average(listeTfr_main)

#il faut recuperer toutes les donnees du trial 1, en faire des averagetfr et mne.grandaverage ça x10
#essai
def gd_average_allEssais(liste_tfr_cond,get_all_suj):
    all_listes = []
    for i in range(10):
        print("essai "+str(i))
        liste_i = []
        for j in range(len(liste_tfr_cond)):#n_sujets
            print("sujet "+str(j))
            if j ==14:
                pass
            else:
                liste_i.append(liste_tfr_cond[j][i].average())
        if get_all_suj:
            all_listes.append(liste_i)
        else:
            all_listes.append(mne.grand_average(liste_i))
    return all_listes

#gd_average1 = mne.grand_average(listei)

all_listes = gd_average_allEssais(listeTfr_main,False)
cond = "main"
for i in range(len(all_listes)):
    all_listes[i].save("../AV_TFR/all_sujets/"+cond+str(i)+"-tfr.h5",overwrite=True)


cond = "main"
all_listes = []
for i in range(10):
  all_listes.append(mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/"+cond+str(i)+"-tfr.h5")[0])

for i in range(10):
    all_listes[i].plot_topomap(fmin=12,fmax=15,tmin=2.5,tmax=26.5)
    
for i in range(10):
    all_listes[i].plot(picks=["C3"],vmin=0,vmax=1e-10,fmax=35,tmin=0,tmax=26.5)

mne.grand_average(all_listes[0:3]).plot_topomap(fmin=12,fmax=15,tmin=2.5,tmax=26.5)
mne.grand_average(all_listes[7:]).plot_topomap(fmin=12,fmax=15,tmin=2.5,tmax=26.5)

(mne.grand_average(all_listes[0:3])-mne.grand_average(all_listes[7:])).plot_topomap(fmin=12,fmax=15,tmin=2.5,tmax=26.5)
raw_signal.plot(block=True)

v = 2.5e-10
vmin = -v
vmax = v
mne.grand_average(all_listes[0:3]).plot(picks=["C3"],fmax=40,mode="logratio",vmin=vmin,vmax=vmax)
mne.grand_average(all_listes[7:]).plot(picks=["C3"],fmax=40,mode="logratio",vmin=vmin,vmax=vmax)
raw_signal.plot(block=True)


mne.grand_average(all_listes[0:3]).plot(baseline=(-3,1),picks=["C3"],fmax=40,mode="logratio",tmin=-3,tmax=26.5)
mne.grand_average(all_listes[7:]).plot(baseline=(-3,1),picks=["C3"],fmax=40,mode="logratio",tmin=-3,tmax=26.5)
raw_signal.plot(block=True)

#========= REFAIRE EN EXCLUANT LES ESSAIS JETES ==========================================

#faire un truc ou on load les data et on fait tourner tout
compte_jetes = [0 for i in range(23)]
def gd_average_allEssais_V2(liste_tfr_cond,get_all_suj,essaisJetes_cond,baseline):
    dureePreBaseline = 3 #3
    dureePreBaseline = - dureePreBaseline
    dureeBaseline = 2.0 #2.0
    valeurPostBaseline = dureePreBaseline + dureeBaseline
    baseline = (dureePreBaseline, valeurPostBaseline)
    all_listes = []
    for i in range(10):
        print("essai "+str(i))
        liste_i = []
        for j in range(len(liste_tfr_cond)):#n_sujets
            print("sujet "+str(j))
            essais_jetes_suj = essaisJetes_cond[j]
            if i+1 not in essais_jetes_suj:
                if baseline:
                    baselined_data = liste_tfr_cond[j][i-compte_jetes[j]].average().apply_baseline(baseline=baseline, mode='zscore', verbose=None)
                    liste_i.append(baselined_data)
                else:
                    liste_i.append(liste_tfr_cond[j][i-compte_jetes[j]].average())
            else:
                compte_jetes[j] += 1 
                print("essai jete")
        if get_all_suj:
            all_listes.append(liste_i)
        else:
            all_listes.append(mne.grand_average(liste_i))
    return all_listes,compte_jetes

def compute_tfr_perTrial(liste_path_cond,jetes_cond,save,namecond,baseline):
    listeTfr_cond_exclude_bad = load_tfr_data_windows(liste_path_cond,"alltrials",True)
    all_listes_wo_bad_v2,compte_jetes = gd_average_allEssais_V2(listeTfr_cond_exclude_bad,False,jetes_cond,True)
    if save:
        for i in range(len(all_listes_wo_bad_v2)):
            if baseline:
                all_listes_wo_bad_v2[i].save("../AV_TFR/all_sujets/"+namecond+str(i)+"bad_excluded"+"_baseline"+"-tfr.h5",overwrite=True)
            else:
                all_listes_wo_bad_v2[i].save("../AV_TFR/all_sujets/"+namecond+str(i)+"bad_excluded"+"-tfr.h5",overwrite=True)
    return all_listes_wo_bad_v2,compte_jetes

# =============

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

all_listes_wo_bad_v2,compte_jetes = compute_tfr_perTrial(liste_rawPathMain,jetes_main,True,"main")
all_listes_wo_bad_v2,compte_jetes = compute_tfr_perTrial(liste_rawPathPendule,jetes_pendule,True,"pendule")
all_listes_wo_bad_v2_mainvib,compte_jetes = compute_tfr_perTrial(liste_rawPathMainIllusion,jetes_mainIllusion,True,"mainIllusion")

all_listes_wo_bad_v2_zscore,compte_jetes = compute_tfr_perTrial(liste_rawPathMain,jetes_main,True,"main",True)
all_listes_wo_bad_v2_zscore,compte_jetes = compute_tfr_perTrial(liste_rawPathPendule,jetes_pendule,True,"pendule",True)

listeTfr_cond_exclude_bad = load_tfr_data_windows(liste_rawPathMainIllusion,"alltrials",True)
all_listes_wo_bad_v2_zscore,compte_jetes = gd_average_allEssais_V2(listeTfr_cond_exclude_bad,False,jetes_mainIllusion,True)
for i in range(len(all_listes_wo_bad_v2_zscore)):
    all_listes_wo_bad_v2_zscore[i].save("../AV_TFR/all_sujets/"+"mainIllusion"+str(i)+"bad_excluded"+"_baseline"+"-tfr.h5",overwrite=True)
        
mne.grand_average(all_listes_wo_bad_v2_zscore[0:3]).plot(picks=["C3"],fmax=40)
mne.grand_average(all_listes_wo_bad_v2_zscore[7:]).plot(picks=["C3"],fmax=40)
raw_signal.plot(block=True)

#LOAD THE DATA WITH ZSCORE BASELINE

cond = "mainIllusion"
all_listes_wo_bad_v2_zscore = []
for i in range(10):
  all_listes_wo_bad_v2_zscore.append(mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/"+cond+str(i)+"bad_excluded"+"_baseline"+"-tfr.h5")[0])

mne.grand_average(all_listes_wo_bad_v2_zscore[0:3]).plot(picks=["C3"],fmax=40,vmin=-1.2,vmax=1.2)
mne.grand_average(all_listes_wo_bad_v2_zscore[7:]).plot(picks=["C3"],fmax=40,vmin=-1.2,vmax=1.2)
raw_signal.plot(block=True)

(mne.grand_average(all_listes_wo_bad_v2_zscore[7:])-mne.grand_average(all_listes_wo_bad_v2_zscore[0:3])).plot(picks=["C3"],fmax=40,vmin=-1.2,vmax=1.2)

(mne.grand_average(all_listes_wo_bad_v2_zscore[7:])-mne.grand_average(all_listes_wo_bad_v2_zscore[0:3])).plot_topomap(fmin=12,fmax=15,tmin=2.5,tmax=26.5)
(mne.grand_average(all_listes_wo_bad_v2_zscore[7:])-mne.grand_average(all_listes_wo_bad_v2_zscore[0:3])).plot_topomap(fmin=8,fmax=30,tmin=2.5,tmax=26.5)
raw_signal.plot(block=True)

for i in range(10):
    all_listes_wo_bad_v2_zscore[i].plot_topomap(fmin=12,fmax=15,tmin=2.5,tmax=26.5,vmin=-0.4,vmax=0.4)
raw_signal.plot(block=True)

#============================================================================
#LOAD THE DATA WITH BAD EXCLUDED (without baseline)
cond = "pendule"
all_listes_wo_bad_v2 = []
for i in range(10):
  all_listes_wo_bad_v2.append(mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/"+cond+str(i)+"bad_excluded"+"-tfr.h5")[0])

(mne.grand_average(all_listes_wo_bad_v2[7:])-mne.grand_average(all_listes_wo_bad_v2[0:3])).plot(picks=["C3"],fmax=40,vmin=0,vmax=2.4e-10)

#(mne.grand_average(all_listes_wo_bad_v2[7:])-mne.grand_average(all_listes_wo_bad_v2[0:3])).plot_topomap(fmin=12,fmax=15,tmin=2.5,tmax=26.5)
(mne.grand_average(all_listes_wo_bad_v2[7:])-mne.grand_average(all_listes_wo_bad_v2[0:3])).plot_topomap(fmin=8,fmax=30,tmin=2.5,tmax=26.5)
raw_signal.plot(block=True)

for i in range(10):
    all_listes_wo_bad_v2[i].plot_topomap(fmin=12,fmax=15,tmin=2.5,tmax=26.5,vmin=0,vmax=2.4e-10)
raw_signal.plot(block=True)


def get_liste_values_cond(cond,baseline):
    liste_wo_bad = []
    for i in range(10):
        if baseline:
            liste_wo_bad.append(mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/"+cond+str(i)+"bad_excluded"+"_baseline"+"-tfr.h5")[0])
        else:
            liste_wo_bad.append(mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/"+cond+str(i)+"bad_excluded"+"-tfr.h5")[0])
    liste_val_freqs = []
    for i in range(10):
        val_essai_i = np.mean(liste_wo_bad[i].data[11][5:28][:,1875:7875])
        print(val_essai_i)
        liste_val_freqs.append(val_essai_i)
    return liste_val_freqs


#raw data
liste_val_freqs_pend = get_liste_values_cond("pendule",False)
liste_val_freqs_main = get_liste_values_cond("main",False)
liste_val_freqs_mainIllusion = get_liste_values_cond("mainIllusion",False)

import matplotlib.pyplot as plt
plt.plot([1,2,3,4,5,6,7,8,9,10],liste_val_freqs_pend,label="pendule")
plt.plot([1,2,3,4,5,6,7,8,9,10],liste_val_freqs_main,label="main")
plt.plot([1,2,3,4,5,6,7,8,9,10],liste_val_freqs_mainIllusion,label="mainIllusion")
plt.legend()
plt.show()

#baseline
liste_val_freqs_pend_bl = get_liste_values_cond("pendule",True)
liste_val_freqs_main_bl = get_liste_values_cond("main",True)
liste_val_freqs_mainIllusion_bl = get_liste_values_cond("mainIllusion",True)
plt.plot([1,2,3,4,5,6,7,8,9,10],liste_val_freqs_pend,label="pendule")
plt.plot([1,2,3,4,5,6,7,8,9,10],liste_val_freqs_main,label="main")
plt.plot([1,2,3,4,5,6,7,8,9,10],liste_val_freqs_mainIllusion,label="mainIllusion")
plt.legend()
plt.show()


#=======

mne.grand_average(all_listes_wo_bad_v2[0:3]).plot(picks=["C3"],fmax=40)
mne.grand_average(all_listes_wo_bad_v2[7:]).plot(picks=["C3"],fmax=40)
raw_signal.plot(block=True)

mne.grand_average(all_listes_wo_bad_v2_mainvib[0:3]).plot(picks=["C3"],fmax=40)
mne.grand_average(all_listes_wo_bad_v2_mainvib[7:]).plot(picks=["C3"],fmax=40)
raw_signal.plot(block=True)

(mne.grand_average(all_listes_wo_bad_v2_mainvib[7:])-mne.grand_average(all_listes_wo_bad_v2_mainvib[0:3])).plot(picks=["C3"],fmax=40)
raw_signal.plot(block=True)
(mne.grand_average(all_listes_wo_bad_v2_mainvib[7:])-mne.grand_average(all_listes_wo_bad_v2_mainvib[0:3])).plot_topomap(fmin=12,fmax=15,tmin=2.5,tmax=26.5)
raw_signal.plot(block=True)


(mne.grand_average(all_listes_wo_bad_v2[7:])-mne.grand_average(all_listes_wo_bad_v2[0:3])).plot(picks=["C3"],fmax=40)
raw_signal.plot(block=True)

(mne.grand_average(all_listes_wo_bad_v2[7:])-mne.grand_average(all_listes_wo_bad_v2[0:3])).plot_topomap(fmin=12,fmax=15,tmin=2.5,tmax=26.5)
raw_signal.plot(block=True)

for i in range(10):
    all_listes_wo_bad_v2[i].plot_topomap(fmin=12,fmax=15,tmin=2.5,tmax=26.5)
raw_signal.plot(block=True)

# listeTfr_main_exclude_bad = load_tfr_data_windows(liste_rawPathMain,"alltrials",True)
# listeTfr_pendule_exclude_bad = load_tfr_data_windows(liste_rawPathPendule,"alltrials",True)
# #listeTfr_mainIllusion_exclude_bad = load_tfr_data_windows(liste_rawPathMain,"alltrials",True)
# all_listes_wo_bad = gd_average_allEssais_V2(listeTfr_pendule_exclude_bad,False)


# dispo_main = []
# for i in range(len(jetes_main)):
#     print("sujet "+str(i))
#     liste = jetes_main[i]
#     liste_dispo_suj = [1,2,3,4,5,6,7,8,9,10]
#     print(liste_dispo_suj)
#     for elt in liste:
#         print(elt)
#         liste_dispo_suj.remove(elt)
#         print(liste_dispo_suj)
#     dispo_main.append(liste_dispo_suj)


# all_listes_wo_bad_v2,compte_jetes = gd_average_allEssais_V2(listeTfr_main_exclude_bad,False,jetes_main)

# cond = "main"
# for i in range(len(all_listes_wo_bad_v2)):
#     all_listes_wo_bad_v2[i].save("../AV_TFR/all_sujets/"+cond+str(i)+"bad_excluded"+"-tfr.h5",overwrite=True)




