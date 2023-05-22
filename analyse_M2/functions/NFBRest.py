#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:02:43 2021

@author: claire.dussard
"""
import os 
import seaborn as sns
import pathlib
import mne
#necessite d'avoir execute handleData_subject.py, et load_savedData avant 
import numpy as np 
#pour se placer dans les donnees lustre
os.chdir("../../../../..")
lustre_data_dir = "cenir/analyse/meeg/BETAPARK/_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

sujetsPb = [11]
for sujetpb in sujetsPb:
    allSujetsDispo.remove(sujetpb)

essaisMainRest = ["9-2" for i in range(25)]
essaisMainIllusionRest = ["10-2" for i in range(25)]
liste_rawPathMain = createListeCheminsSignaux(essaisMainRest,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusionRest,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale)

event_id_mainIllusion = {'Essai_mainIllusion':3}
event_id_main={'Essai_main':3}  


#=================================================================================================================================
                                                        #METHODE AVERAGE EPOCHS SUJETS
#=================================================================================================================================

nbSujets = 24
SujetsDejaTraites = 0
rawPath_main_sujets = liste_rawPathMain[SujetsDejaTraites:SujetsDejaTraites+nbSujets]
rawPath_mainIllusion_sujets = liste_rawPathMainIllusion[SujetsDejaTraites:SujetsDejaTraites+nbSujets]


listeEpochs_main,listeICA_main,listeEpochs_mainIllusion,listeICA_mainIllusion = all_conditions_analysis_NFBRest(allSujetsDispo,rawPath_main_sujets,rawPath_mainIllusion_sujets,
                            event_id_main,event_id_mainIllusion,
                            0.1,1,90,[50,100],'Fz')

#sujets 15 et 16 a jeter / verif raw : vagues partout

saveEpochsAfterICA(listeEpochs_main,rawPath_main_sujets)
save_ICA_files(listeICA_main,rawPath_main_sujets)
saveEpochsAfterICA(listeEpochs_mainIllusion,rawPath_mainIllusion_sujets)
save_ICA_files(listeICA_mainIllusion,rawPath_mainIllusion_sujets)

EpochDataMain = load_data_postICA(rawPath_main_sujets,"")

EpochDataMainIllusion = load_data_postICA(rawPath_mainIllusion_sujets,"")

#===================set montage===IMPORTANT!!!!=======================
montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
for epochs in EpochDataMain:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)
for epochs in EpochDataMainIllusion:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)
        
        
liste_power_main = plotSave_power_topo_cond(EpochDataMain,rawPath_main_sujets,3,85,"NFBrest_main",250.,1.5,25.5)#needs to have set up the electrode montage before
liste_power_mainIllusion = plotSave_power_topo_cond(EpochDataMainIllusion,rawPath_mainIllusion_sujets,3,85,"NFBrest_mainIllusion",250.,1.5,25.5)

save_tfr_data(liste_power_main,rawPath_main_sujets,"")

save_tfr_data(liste_power_mainIllusion,rawPath_mainIllusion_sujets,"")

#===================apply a baseline by subject before grand averaging=========================

dureePreBaseline = 4
dureePreBaseline = - dureePreBaseline
dureeBaseline = 3.0
valeurPostBaseline = dureePreBaseline + dureeBaseline

baseline = (dureePreBaseline, valeurPostBaseline)
for tfr in liste_power_mainIllusion:
    tfr.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    
av_power_main_rest = mne.grand_average(liste_power_main,interpolate_bads=True)
save_topo_data(av_power_main_rest,dureePreBaseline,valeurPostBaseline,"all_sujets",mode,"NFBrest_main",False,1.5,25.5)

av_power_mainIllusion_rest = mne.grand_average(liste_power_mainIllusion,interpolate_bads=True)
save_topo_data(av_power_mainIllusion_rest,dureePreBaseline,valeurPostBaseline,"all_sujets",mode,"NFBrest_mainIllusion",False,1.5,25.5)

av_power_main_rest.save("../AV_TFR/all_sujets/main_NFBrest-tfr.h5",overwrite=True)
av_power_mainIllusion_rest.save("../AV_TFR/all_sujets/mainIllusion_NFBrest-tfr.h5",overwrite=True)
#========== compute difference between conditions ========================
avpower_main_moins_mainIllusion = av_power_main_rest - av_power_mainIllusion_rest

save_topo_data(avpower_main_moins_mainIllusion,dureePreBaseline,valeurPostBaseline,"all_sujets",mode,"NFBrest_main-mainIllusion",False,1.5,25.5)