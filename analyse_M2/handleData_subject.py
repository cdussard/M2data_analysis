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
     