#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 10:57:01 2021

@author: claire.dussard
"""
import pandas as pd
os.chdir("../../../../..")
lustre_data_dir = "cenir/analyse/meeg/BETAPARK/_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

channelsSansFz = ['Fp1', 'Fp2', 'F7', 'F3','F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10','T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5','CP1','CP2','CP6','TP10','P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2','HEOG','VEOG']

dict_numSujet_posPremiersSujets =	{
  0 : 0 ,
  2 : 1,
  3 : 2,
  5 : 3,
}

#functions useful to review raw data before filtering
def review_all_NFB_data():
    liste_rawPath_main = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale)
    liste_rawPath_mainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale)
    liste_rawPath_pendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale)
    event_id_mainIllusion = {'Essai_mainIllusion':3}
    event_id_pendule={'Essai_pendule':4}  
    event_id_main={'Essai_main':3}  
    for numSujet in range(25):
        print("sujet n°"+str(numSujet))
        review_one_subj_one_cond_data(numSujet,liste_rawPath_main,event_id_main)
        review_one_subj_one_cond_data(numSujet,liste_rawPath_mainIllusion,event_id_mainIllusion)
        review_one_subj_one_cond_data(numSujet,liste_rawPath_pendule,event_id_pendule)
        
#review_all_NFB_data()     

def review_one_subj_one_cond_data(numSujet,liste_rawPath_cond,event_id,channelsSansFz):
    if numSujet in allSujetsDispo:
        if numSujet<6:
            print("correcting subject position")
            posSujet = dict_numSujet_posPremiersSujets[numSujet]#conversion pour avoir la position du fichier du sujet (suj 4 et 1 manquants)
        else:
            posSujet = numSujet - 2
        extension_nom_fichier = liste_rawPath_cond[posSujet]
        print(extension_nom_fichier)
        print("reading raw file...")
        raw_fichier_sujet_cond = mne.io.read_raw_brainvision(extension_nom_fichier,preload=False,eog=('HEOG', 'VEOG'))
        #raw_fichier_sujet_cond.plot(n_channels = 40,block=True)
        eeg_emg_eogFile = raw_fichier_sujet_cond.drop_channels(['ECG' ,'ACC_X','ACC_Z','ACC_Y','EMG'])#,'HEOG','VEOG'])'EMG'
        #plotting raw file
        #eeg_emg_eogFile.plot(n_channels = 35,block=True)
        #plotting NFB trials
        events = mne.events_from_annotations(eeg_emg_eogFile)[0]
        print(events)
        epochsNFB = mne.Epochs(eeg_emg_eogFile,events,event_id,tmin=-5,tmax = 31,baseline=None, preload=True)
        epochsNFB.reorder_channels(channelsSansFz)
        epochsNFB.plot(block=True,n_channels=35)
    else:
        print("numSujet not in sujets Dispo")

review_one_subj_one_cond_data(24,liste_rawPathMainIllusion,event_id_mainIllusion,channelsSansFz)

review_one_subj_one_cond_data(6,liste_rawPathMain,event_id_main,channelsSansFz)

#=============load tfr data and check time frequency topo==============================
tfr_sujet9 = load_tfr_data(rawPath_mainIllusion_sujets[8:9],"")[0]

tfr_sujet9.plot(block=True)

review_one_subj_one_cond_data(2,rawPath_mainIllusion_sujets,event_id_mainIllusion)
review_one_subj_one_cond_data(0,liste_rawPathMain,event_id_main,channelsSansFz)

#check data after bad have been dropped
num_sujet = 1
# EpochDataMain = load_data_postICA_postdropBad(rawPath_main_sujets[num_sujet:num_sujet+1],"")

# EpochDataPendule = load_data_postICA_postdropBad(rawPath_pendule_sujets[num_sujet:num_sujet+1],"")

# EpochDataMainIllusion = load_data_postICA_postdropBad(rawPath_mainIllusion_sujets[num_sujet:num_sujet+1],"")


#=============load tfr data and check time frequency topo==============================
tfr_sujet2 = load_tfr_data(rawPath_mainIllusion_sujets[2:3],"")
tfr_sujet2[0].plot_topo(baseline=None, mode='logratio', title='Average power',fmin=8,fmax=30)

raw_signal.plot(block=True)#,event_id=event_dict)

raw_fichier_sujet_cond = review_one_subj_one_cond_data(6,liste_rawPathMainIllusion,event_id_mainIllusion)

# stats_electrodes_main = pd.read_csv("csv_files/stats_epochs/stats_epochs_sujets_main.csv")
#============= review TFR data average conditions===================================
av_power_main =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main-tfr.h5")
av_power_mainIllusion =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion-tfr.h5")

av_power_pendule =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule-tfr.h5")

av_power_mainIllusion[0].plot_topo(baseline=None, mode='logratio', title='Average power',fmin=3,fmax=80)

#EpochDataMain_s06 = load_data_postICA(liste_rawPath_main[4:5],"") 


#=================== load data test corrige Laurent==============
#test donnees sur 1 sujet

num_sujet = 16
sample_data_loc = listeNumSujetsFinale[num_sujet]+"/"+listeDatesFinale[num_sujet]+"/eeg"
sample_data_dir = pathlib.Path(sample_data_loc)

raw_path_sample_initial = sample_data_dir/("BETAPARK_2021-05-05_6-2.vhdr")
raw_signal_corrige = mne.io.read_raw_brainvision(raw_path_sample_initial,preload=False,eog=('HEOG', 'VEOG'))
raw_signal_corrige.plot(block=True)