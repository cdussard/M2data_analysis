#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 12:21:11 2021

@author: claire.dussard
"""
import os 
import seaborn as sns
import pathlib
#recuperation des donnees sujet IM seule
#necessite d'avoir execute handleData_subject.py avant 
import numpy as np 
#pour se placer dans les donnees lustre
os.chdir("../../../../..")
lustre_data_dir = "cenir/analyse/meeg/BETAPARK/_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

nom_essais = ["1-1","1-2"]#on commence avec les deux fichiers
essaisBaseline1 = ["1-1" for i in range(25)]
essaisBaseline2 = ["1-2" for i in range(25)]

liste_rawPathBaseline1 = createListeCheminsSignaux(essaisBaseline1,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale)
liste_rawPathBaseline2 = createListeCheminsSignaux(essaisBaseline2,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale)

event_id_baseline={'Début baseline rest':12}

#=================================================================================================================================
                                                        #METHODE AVERAGE EPOCHS SUJETS
#=================================================================================================================================
nbSujets = 2
SujetsDejaTraites = 6
rawPath_baseline_sujets1 = liste_rawPathBaseline1[SujetsDejaTraites:SujetsDejaTraites+nbSujets]
rawPath_baseline_sujets2 = liste_rawPathBaseline2[SujetsDejaTraites:SujetsDejaTraites+nbSujets]

listeEpochs_baseline,listeICA_baseline = baselineRest_analysis(allSujetsDispo,rawPath_baseline_sujets1,rawPath_baseline_sujets2,
                            event_id_baseline,0.1,1,90,[50,100],'Fz')

#sauvegarder les epochs

tousEpochs_sujet0 = mne.epochs.concatenate_epochs(listeEpochs_baseline[0])#15 * 2 epochs
tousEpochs_sujet0.plot_psd(1,50,estimate="power",picks=['C3','C4','Cz'])

listeEpochs_baseline[1][1].info["bads"]=['TP9']
tousEpochs_sujet1 = mne.epochs.concatenate_epochs(listeEpochs_baseline[1])#15 * 2 epochs
tousEpochs_sujet1.plot_psd(1,50,estimate="power",picks=['C3','C4','Cz'])

raw_signal.plot(block=True)

#listeEpochs_baseline,listeICA_baseline 
#==================================================================================================================================

# SujetsPbNomFichiers = [9]#9 : manque recording (on a l'OV)
# SujetsExclusAnalyse = [1,4]
# SujetsAvecPb = np.unique(SujetsPbNomFichiers + SujetsExclusAnalyse)
# allSujetsDispo = [i for i in range(nombreTotalSujets+1)]
# for sujet_pbmatique in SujetsAvecPb:  
#     allSujetsDispo.remove(sujet_pbmatique)
# print(len(allSujetsDispo))

# liste_rawPath = []
# for num_sujet in allSujetsDispo:
#     print("sujet n° "+str(num_sujet))
#     nom_sujet = listeNumSujetsFinale[num_sujet]
#     date_nom_fichier = dates[num_sujet][-4:]+"-"+dates[num_sujet][3:5]+"-"+dates[num_sujet][0:2]+"_"
#     dateSession = listeDatesFinale[num_sujet]
#     sample_data_loc = listeNumSujetsFinale[num_sujet]+"/"+listeDatesFinale[num_sujet]+"/eeg"
#     sample_data_dir = pathlib.Path(sample_data_loc)
#     for nom_essai in nom_essais:
#         raw_path_sample = sample_data_dir/("BETAPARK_"+ date_nom_fichier + nom_essai+".vhdr")
#         liste_rawPath.append(raw_path_sample)

# print(liste_rawPath)#peut etre qu'il vaut mieux les separer par sujet ?

# listeAverageRef,listeRaw = pre_process_donnees(liste_rawPath[2:],1,80,[50,100],31,'Fz',False,[])
# listeICApreproc,listeICA = ICA_preproc(listeAverageRef,listeRaw,[],31,98,False)

# events = mne.events_from_annotations(listeICApreproc[1])

# event_id={'Début baseline rest':12}

# events = mne.events_from_annotations(listeICApreproc[0])

# liste_tous_epochs = []
# for numSignal in range(len(listeICApreproc)):
#     if numSignal == 7 or numSignal ==9 or numSignal ==12 or numSignal ==14:#marqueurs pas enregistres :(
#         pass
#     else:
#         print(numSignal)
#         events = mne.events_from_annotations(listeICApreproc[numSignal])[0]#baseline=(-2, 0)
#         print (events)
#         signal = listeICApreproc[numSignal]
#         epochsCibles = mne.Epochs(signal,events,event_id,tmin=-4.0,tmax = 120.0,baseline=None)
#         liste_tous_epochs.append(epochsCibles)

# tousEpochs.plot(block=True)#verif visuelle voire exclusion si besoin
# tousEpochs = mne.epochs.concatenate_epochs(liste_tous_epochs)#15 * 2 epochs
# tousEpochs.save("2minBaselineRest_11suj_epo.fif") #loading point :)

# tousEpochs = mne.read_epochs("2minBaselineRest_11suj_epo.fif") #loading point 

# tousEpochs.plot_psd(1,50,estimate="power",picks=['C3','C4','Cz'])

# #pour debloquer les graphs
# raw_signal.plot(block=True)

# #calculer le marqueur de Blankertz cf code Camille Benaroch