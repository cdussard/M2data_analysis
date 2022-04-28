#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 10:36:12 2021

@author: claire.dussard
"""

#imports 
import pathlib
import mne
import os
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np

def createSujetsData():
    import pandas as pd
    import numpy as np
    #repertoire depart : cd Bureau/Scripts
    #cd analyse_M2
    #repertoire donnees sujet
    donneesSujet = pd.read_csv("data/generalData.csv")
    seuils_sujets = pd.read_csv("./data/seuil_data/seuils_sujets_dash.csv")
    
    essaisMainSeule = donneesSujet.refaitMainSeule
    essaisPendule = donneesSujet.refaitPendule
    essaisMainTactile = donneesSujet.refaitMainTactile
    essaisMainIllusion = donneesSujet.refaitMainIllusion
    
    #recuperation dates experiences 
    dates = donneesSujet["date"]
    
    #creation noms fichiers sujets
    nombreTotalSujets = 24
    SujetsPbNomFichiers = []
    if nombreTotalSujets >10:
       listeNumSujets = ["00"+str(num) for num in range (1,10)]
       listeNumSujets.extend(["0"+str(num) for num in range (10,nombreTotalSujets+1)])
    else:
        listeNumSujets = ["00"+str(num) for num in range (nombreTotalSujets)]
    
    listeNumSujetsFinale = ["S" + nom for nom in listeNumSujets] 
    listeNumSujetsFinale.insert(0,"Pilote2")
    listeNumSujetsFinale = ["sub-" + nom for nom in listeNumSujetsFinale]
    
    listeDatesFinale = []
    if len(dates == nombreTotalSujets):
        for date in dates:
            dateReformatee = date[-4:] + date[3:5] + date[0:2]
            listeDatesFinale.append(dateReformatee)
        listeDatesFinale = ["ses-" + date for date in listeDatesFinale]
    else:
        print("il manque des dates")
        
    #lecture des fichiers 
    SujetsPbNomFichiers = [0,1,2,3,4,5,6]#ne pas supprimer, permet de corriger les dates dans la fonction qui cree les chemins de fichier
    SujetsExclusAnalyse = [1,4]
    SujetsPbEnregistrementCondition = []#[10]#,22]
    Sujet3conditions = [0,3,5,6,8,12,13,14,16,17,20,21,22,23,24]
    Sujets2conditions = [2,7,9,10,11,15,18,19]
    SujetsAvecPb = np.unique(SujetsExclusAnalyse)
    allSujetsDispo = [i for i in range(nombreTotalSujets+1)]
    for sujet_pbmatique in SujetsAvecPb:#zip(SujetsAvecPb,SujetsPbEnregistrementCondition):  
        allSujetsDispo.remove(sujet_pbmatique) #allSujetsDispo.remove(sujetPbRecord)

    return essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets
     
essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData() 


def createSujetsData_obsExec():
    import pandas as pd
    import numpy as np
    #repertoire depart : cd Bureau/Scripts
    #cd analyse_M2
    #repertoire donnees sujet
    donneesSujet = pd.read_csv("data/generalData.csv")
    seuils_sujets = pd.read_csv("./data/seuil_data/seuils_sujets_dash.csv")
    
    essaisObsExecIm = donneesSujet.obsExecIm
    
    #recuperation dates experiences 
    dates = donneesSujet["date"]
    
    #creation noms fichiers sujets
    nombreTotalSujets = 24
    SujetsPbNomFichiers = []
    if nombreTotalSujets >10:
       listeNumSujets = ["00"+str(num) for num in range (1,10)]
       listeNumSujets.extend(["0"+str(num) for num in range (10,nombreTotalSujets+1)])
    else:
        listeNumSujets = ["00"+str(num) for num in range (nombreTotalSujets)]
    
    listeNumSujetsFinale = ["S" + nom for nom in listeNumSujets] 
    listeNumSujetsFinale.insert(0,"Pilote2")
    listeNumSujetsFinale = ["sub-" + nom for nom in listeNumSujetsFinale]
    
    listeDatesFinale = []
    if len(dates == nombreTotalSujets):
        for date in dates:
            dateReformatee = date[-4:] + date[3:5] + date[0:2]
            listeDatesFinale.append(dateReformatee)
        listeDatesFinale = ["ses-" + date for date in listeDatesFinale]
    else:
        print("il manque des dates")
        
    #lecture des fichiers 
    SujetsPbNomFichiers = [0,1,2,3,4,5,6]#ne pas supprimer, permet de corriger les dates dans la fonction qui cree les chemins de fichier
    SujetsExclusAnalyse = [1,4]
    SujetsPbEnregistrementCondition = []#[10]#,22]
    Sujet3conditions = [0,3,5,6,8,12,13,14,16,17,20,21,22,23,24]
    Sujets2conditions = [2,7,9,10,11,15,18,19]
    SujetsAvecPb = np.unique(SujetsExclusAnalyse)
    allSujetsDispo = [i for i in range(nombreTotalSujets+1)]
    for sujet_pbmatique in SujetsAvecPb:#zip(SujetsAvecPb,SujetsPbEnregistrementCondition):  
        allSujetsDispo.remove(sujet_pbmatique) #allSujetsDispo.remove(sujetPbRecord)

    return essaisObsExecIm,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets
     
#=========================test donnees sur 1 sujet=============================================================================
# num_sujet = 19
# sample_data_loc = listeNumSujetsFinale[num_sujet]+"/"+listeDatesFinale[num_sujet]+"/eeg"
# sample_data_dir = pathlib.Path(sample_data_loc)
# date_nom_fichier = dates[num_sujet][-4:]+"-"+dates[num_sujet][3:5]+"-"+dates[num_sujet][0:2]
# raw_path_sample = sample_data_dir/("BETAPARK_"+ date_nom_fichier + "_7-2-b.vhdr")

# raw_signal = mne.io.read_raw_brainvision(raw_path_sample,preload=False,eog=('HEOG', 'VEOG'))

# raw_signal.plot(block=True)#,event_id=event_dict)
# print(raw_signal.info)
# events = mne.events_from_annotations(raw_signal)
# #get events
# event_dict = {
#         'debut trial':4,
#         'fin du trial':6,
#         'exp start':10,
#         'croix debut':12,
#         'fin croix':14,
#         'instruction imaginer':18,
#         'instruction repos':19,
#         'mvt main':22, 
#         'debut FB':27,
#         #'aucune idee':240,
#         'start':99999,}

#         #'OV start':255}

# raw_signal.plot(events[0],block=True)#,event_id=event_dict)#il les affiche par dessus :')

# afficher les events
# mne.viz.plot_events(events[0], sfreq=raw_signal.info['sfreq'],first_samp=raw_signal.first_samp, event_id = event_dict)   

