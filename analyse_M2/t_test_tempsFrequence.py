#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 11:08:18 2022

@author: claire.dussard
"""
import pandas as pd
import scipy
#================= get the data===============
from functions.load_savedData import *

from handleData_subject import createSujetsData
from functions.load_savedData import *
from frequencyPower_displays import *

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates = createSujetsData()

seuils_sujets = pd.read_csv("./data/seuil_data/seuils_sujets_dash.csv")

#pour se placer dans les donnees lustre
os.chdir("../../../../../../")
lustre_data_dir = "iss02/cenir/analyse/meeg/BETAPARK/_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)


liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

nbSujets = 24
SujetsDejaTraites = 0
rawPath_main_sujets = liste_rawPathMain[SujetsDejaTraites:SujetsDejaTraites+nbSujets]
rawPath_pendule_sujets = liste_rawPathPendule[SujetsDejaTraites:SujetsDejaTraites+nbSujets]
rawPath_mainIllusion_sujets = liste_rawPathMainIllusion[SujetsDejaTraites:SujetsDejaTraites+nbSujets]
#================================= T -TEST AVEC BASELINE ==============================================
#C3 8-30Hz, 23 sujets, 3 conditions pour commencer

liste_tfrMain = load_tfr_data(rawPath_main_sujets,"")
liste_tfrMainIllusion = load_tfr_data(rawPath_mainIllusion_sujets,"")
liste_tfrPendule = load_tfr_data(rawPath_pendule_sujets,"")


liste_tfr_mainIllusion = liste_tfrMainIllusion.copy()
liste_tfr_main = liste_tfrMain.copy()
liste_tfr_pendule = liste_tfrPendule.copy()
#compute baseline
baseline = (-3,-1)
for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
    tfr_mainI.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    tfr_pendule.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    tfr_main.apply_baseline(baseline=baseline, mode='logratio', verbose=None)

#crop time & frequency
for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
    tfr_mainI.crop(tmin = 1.5,tmax=25.5,fmin = 8,fmax = 30)
    tfr_main.crop(tmin = 1.5,tmax=25.5,fmin = 8,fmax = 30)
    tfr_pendule.crop(tmin = 1.5,tmax=25.5,fmin = 8,fmax = 30)
#subset electrode
for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
    tfr_mainI.pick_channels(["C3"])
    tfr_main.pick_channels(["C3"])
    tfr_pendule.pick_channels(["C3"])
#apply seuil NE PAS FAIRE BASELINE DU COUP

#create t test table , with time and frequency
tableauMain = np.zeros(shape=(23,24,23))#freqs x timepoints x sujets
tableauMainIllusion = np.zeros(shape=(23,24,23))#freqs x timepoints x sujets
tableauPendule = np.zeros(shape=(23,24,23))#freqs x timepoints x sujets
for i in range(23):
    print(i)
    pendule_2dim = liste_tfr_pendule[i].data.mean(axis=0)#faire pour autres aussi
    print(pendule_2dim.shape)
    main_2dim = liste_tfr_main[i].data.mean(axis=0)#faire pour autres aussi
    mainI_2dim = liste_tfrMainIllusion[i].data.mean(axis=0)#faire pour autres aussi

    C3_mov_pendule = computeSlidingWindowEpochs(pendule_2dim,24)#a sauvegarder
    print(C3_mov_pendule)#probleme : on a que le temps, pas les frequences ?
    print(C3_mov_pendule.shape)
    C3_mov_main = computeSlidingWindowEpochs(main_2dim,24)
    C3_mov_mainIllusion = computeSlidingWindowEpochs(mainI_2dim,24)


    tableauPendule[:,:,i] = C3_mov_pendule
    tableauMain[:,:,i] = C3_mov_main
    tableauMainIllusion[:,:,i] = C3_mov_mainIllusion

np.save('../numpy_files/C3_mov_pendule_8-30_23sujets.npy', tableauPendule)
np.save('../numpy_files/C3_mov_main_8-30_23sujets.npy', tableauMain)
np.save('../numpy_files/C3_mov_mainIllusion_8-30_23sujets.npy', tableauMainIllusion)

tableauPendule = np.load('../numpy_files/C3_mov_pendule_8-30_23sujets.npy')
tableauMain = np.load('../numpy_files/C3_mov_main_8-30_23sujets.npy')
tableauMainIllusion = np.load('../numpy_files/C3_mov_mainIllusion_8-30_23sujets.npy')

#save as excel
writer = pd.ExcelWriter('../excel_files/avec_BL/C3_mov_pendule_8-30_23sujets.xlsx', engine='xlsxwriter')
for i in range(23):
    df = pd.DataFrame(tableauPendule[:,:,i])
    df.to_excel(writer,index=False,header=False, sheet_name='bin%d' % i)
writer.save()
writer = pd.ExcelWriter('../excel_files/avec_BL/C3_mov_main_8-30_23sujets.xlsx', engine='xlsxwriter')
for i in range(23):
    df = pd.DataFrame(tableauMain[:,:,i])
    df.to_excel(writer,index=False,header=False, sheet_name='bin%d' % i)
writer.save()
writer = pd.ExcelWriter('../excel_files/avec_BL/C3_mov_mainIllusion_8-30_23sujets.xlsx', engine='xlsxwriter')
for i in range(23):
    df = pd.DataFrame(tableauMainIllusion[:,:,i])
    df.to_excel(writer,index=False,header=False, sheet_name='bin%d' % i)
writer.save()


#=============get the data in the right format =========================
tableauMain = pd.ExcelFile('../excel_files/avec_BL/C3_mov_main_8-30_23sujets.xlsx')
tableauPendule = pd.ExcelFile('../excel_files/avec_BL/C3_mov_pendule_8-30_23sujets.xlsx')
tableauMainIllusion = pd.ExcelFile('../excel_files/avec_BL/C3_mov_mainIllusion_8-30_23sujets.xlsx')

listeSujetsMain = []
listeSujetsPendule = []
listeSujetsMainIllusion = []

for i in range(23):
    df_Sujet_i_Main = pd.read_excel(tableauMain, "bin"+str(i),header=None)
    listeSujetsMain.append(df_Sujet_i_Main)
    
    df_Sujet_i_Pendule = pd.read_excel(tableauPendule, "bin"+str(i),header=None)
    listeSujetsPendule.append(df_Sujet_i_Pendule)
    
    df_Sujet_i_MainIllusion = pd.read_excel(tableauMainIllusion, "bin"+str(i),header=None)
    listeSujetsMainIllusion.append(df_Sujet_i_MainIllusion)
    
    
def get_t_testValue_cell(ligne,colonne):
    print("ligne : "+str(ligne))
    print("colonne : "+str(colonne))
    ligne = ligne
    colonne = colonne
    listeValeurs_cellule_main = []
    listeValeurs_cellule_pendule = []
    listeValeurs_cellule_mainIllusion = []
    
    for i in range(23):
        listeValeurs_cellule_main.append(listeSujetsMain[i].iloc[ligne,colonne])
        listeValeurs_cellule_pendule.append(listeSujetsPendule[i].iloc[ligne,colonne])
        listeValeurs_cellule_mainIllusion.append(listeSujetsMainIllusion[i].iloc[ligne,colonne])

    resultatMainPendule = scipy.stats.ttest_rel(listeValeurs_cellule_main,listeValeurs_cellule_pendule)
    resultatMainMainIllusion = scipy.stats.ttest_rel(listeValeurs_cellule_main,listeValeurs_cellule_mainIllusion)

    return resultatMainPendule,resultatMainMainIllusion


#pour toutes les valeurs
nbFreq = listeSujetsMain[0].shape[0] #lire le nb de points de frequence
nbPointsTemps = listeSujetsMain[0].shape[1]#lire le nb de points de temps
tableauValeurs_mainPendule = np.zeros(shape=(nbFreq,nbPointsTemps))
tableauValeurs_mainMainIllusion = np.zeros(shape=(nbFreq,nbPointsTemps))
for timePoint in range(nbPointsTemps):#colonne
    for freqPoint in range(nbFreq):#ligne
        res_mainPendule,res_mainMainI = get_t_testValue_cell(freqPoint,timePoint)
        tableauValeurs_mainPendule[freqPoint,timePoint] = res_mainPendule[1]#p value
        tableauValeurs_mainMainIllusion[freqPoint,timePoint] = res_mainMainI[1]#p value
        
    
pd.DataFrame(tableauValeurs_mainPendule).to_csv("../excel_files/avec_BL/t_test_23sujets_tpsFreq_mainPendule.csv",header=None, index=None)
pd.DataFrame(tableauValeurs_mainMainIllusion).to_csv("../excel_files/avec_BL/t_test_23sujets_tpsFreq_mainMainI.csv",header=None, index=None)



#======================================================================================

#==================== T-TEST ELECTRODE x FREQUENCE ======================================
liste_tfrMain = load_tfr_data(rawPath_main_sujets,"")
liste_tfrMainIllusion = load_tfr_data(rawPath_mainIllusion_sujets,"")
liste_tfrPendule = load_tfr_data(rawPath_pendule_sujets,"")


liste_tfr_mainIllusion = liste_tfrMainIllusion.copy()
liste_tfr_main = liste_tfrMain.copy()
liste_tfr_pendule = liste_tfrPendule.copy()
#compute baseline
baseline = (-3,-1)
for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
    tfr_mainI.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    tfr_pendule.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    tfr_main.apply_baseline(baseline=baseline, mode='logratio', verbose=None)

#crop time & frequency
for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
    tfr_mainI.crop(tmin = 1.5,tmax=25.5,fmin = 8,fmax = 30)
    tfr_main.crop(tmin = 1.5,tmax=25.5,fmin = 8,fmax = 30)
    tfr_pendule.crop(tmin = 1.5,tmax=25.5,fmin = 8,fmax = 30)


#create t test table , with electrodes and frequency
tableauMain = np.zeros(shape=(28,23,23))#electrodes x freqs x sujets
tableauMainIllusion = np.zeros(shape=(28,23,23))#electrodes x freqs x sujets
tableauPendule = np.zeros(shape=(28,23,23))#electrodes x freqs x sujets
for i in range(23):
    allElec_mov_pendule = computeSlidingWindowEpochs_elecFreq(liste_tfr_pendule[i].data,24)
    allElec_mov_main = computeSlidingWindowEpochs_elecFreq(liste_tfr_main[i].data,24)
    allElec_mov_mainIllusion = computeSlidingWindowEpochs_elecFreq(liste_tfr_mainIllusion[i].data,24)
    #moyenner les 24 points de temps
    allElec_mov_pendule_timePooled = allElec_mov_pendule.mean(axis=2)
    allElec_mov_main_timePooled = allElec_mov_main.mean(axis=2)
    allElec_mov_mainIllusion_timePooled = allElec_mov_mainIllusion.mean(axis=2)
    

    tableauPendule[:,:,i] = allElec_mov_pendule_timePooled
    tableauMain[:,:,i] = allElec_mov_main_timePooled
    tableauMainIllusion[:,:,i] = allElec_mov_mainIllusion_timePooled

np.save('../numpy_files/allElec_mov_pendule_timePooled_8-30_23sujets.npy', tableauPendule)
np.save('../numpy_files/allElec_mov_main_timePooled_8-30_23sujets.npy', tableauMain)
np.save('../numpy_files/allElec_mov_mainIllusion_timePooled_8-30_23sujets.npy', tableauMainIllusion)

tableauPendule = np.load('../numpy_files/allElec_mov_pendule_timePooled_8-30_23sujets.npy')
tableauMain = np.load('../numpy_files/allElec_mov_main_timePooled_8-30_23sujets.npy')
tableauMainIllusion = np.load('../numpy_files/allElec_mov_mainIllusion_timePooled_8-30_23sujets.npy')

#save as excel
writer = pd.ExcelWriter('../excel_files/allElec_mov_pendule_timePooled_8-30_23sujets.xlsx', engine='xlsxwriter')
for i in range(23):
    df = pd.DataFrame(tableauPendule[:,:,i])
    df.to_excel(writer,index=False,header=False, sheet_name='bin%d' % i)
writer.save()
writer = pd.ExcelWriter('../excel_files/allElec_mov_main_timePooled_8-30_23sujets.xlsx', engine='xlsxwriter')
for i in range(23):
    df = pd.DataFrame(tableauMain[:,:,i])
    df.to_excel(writer,index=False,header=False, sheet_name='bin%d' % i)
writer.save()
writer = pd.ExcelWriter('../excel_files/allElec_mov_mainIllusion_timePooled_8-30_23sujets.xlsx', engine='xlsxwriter')
for i in range(23):
    df = pd.DataFrame(tableauMainIllusion[:,:,i])
    df.to_excel(writer,index=False,header=False, sheet_name='bin%d' % i)
writer.save()


#=============get the data in the right format =========================
tableauMain = pd.ExcelFile('../excel_files/avec_BL/All_elecs/allElec_mov_main_timePooled_8-30_23sujets.xlsx')
tableauPendule = pd.ExcelFile('../excel_files/avec_BL/All_elecs/allElec_mov_pendule_timePooled_8-30_23sujets.xlsx')
tableauMainIllusion = pd.ExcelFile('../excel_files/avec_BL/All_elecs/allElec_mov_mainIllusion_timePooled_8-30_23sujets.xlsx')

listeSujetsMain = []
listeSujetsPendule = []
listeSujetsMainIllusion = []

for i in range(23):
    df_Sujet_i_Main = pd.read_excel(tableauMain, "bin"+str(i),header=None)#,index=False)
    listeSujetsMain.append(df_Sujet_i_Main)
    
    df_Sujet_i_Pendule = pd.read_excel(tableauPendule, "bin"+str(i),header=None)
    listeSujetsPendule.append(df_Sujet_i_Pendule)
    
    df_Sujet_i_MainIllusion = pd.read_excel(tableauMainIllusion, "bin"+str(i),header=None)
    listeSujetsMainIllusion.append(df_Sujet_i_MainIllusion)
    
    
def get_t_testValue_cell(ligne,colonne):
    print("ligne : "+str(ligne))
    print("colonne : "+str(colonne))
    ligne = ligne
    colonne = colonne
    listeValeurs_cellule_main = []
    listeValeurs_cellule_pendule = []
    listeValeurs_cellule_mainIllusion = []
    
    for i in range(23):
        listeValeurs_cellule_main.append(listeSujetsMain[i].iloc[ligne,colonne])#l'indexage est base sur les noms de colonne
        listeValeurs_cellule_pendule.append(listeSujetsPendule[i].iloc[ligne,colonne])#, ne marche plus quand on vire le header
        listeValeurs_cellule_mainIllusion.append(listeSujetsMainIllusion[i].iloc[ligne,colonne])
    resultatMainPendule = scipy.stats.ttest_rel(listeValeurs_cellule_main,listeValeurs_cellule_pendule)#TO DO : ajouter l'ANOVA
    resultatMainMainIllusion = scipy.stats.ttest_rel(listeValeurs_cellule_main,listeValeurs_cellule_mainIllusion)

    return resultatMainPendule,resultatMainMainIllusion


#pour toutes les valeurs
nbElec = listeSujetsMain[0].shape[0] #lire le nb d'electrodes'
nbFreq = listeSujetsMain[0].shape[1]#lire le nb de frequences
tableauValeurs_mainPendule = np.zeros(shape=(nbElec,nbFreq))
tableauValeurs_mainMainIllusion = np.zeros(shape=(nbElec,nbFreq))
for freqPoint in range(nbFreq):#colonne
    for elec in range(nbElec):#ligne
        print(elec)
        res_mainPendule,res_mainMainI = get_t_testValue_cell(elec,freqPoint)
        tableauValeurs_mainPendule[elec,freqPoint] = res_mainPendule[1]#p value
        tableauValeurs_mainMainIllusion[elec,freqPoint] = res_mainMainI[1]#p value
        
    

pd.DataFrame(tableauValeurs_mainPendule).to_csv("../excel_files/avec_BL/All_elecs/t_test_23sujets_elecFreq_mainPendule.csv",header=None, index=None)
pd.DataFrame(tableauValeurs_mainMainIllusion).to_csv("../excel_files/avec_BL/All_elecs/t_test_23sujets_elecFreq_mainMainI.csv",header=None, index=None)
#==========FDR correction==========
tableauValeurs_mainPendule = pd.read_csv("../excel_files/avec_BL/All_elecs/t_test_23sujets_elecFreq_mainPendule.csv",header=None)

FDR_corrected_mainPendule = mne.stats.fdr_correction(tableauValeurs_mainPendule, alpha=0.05, method='indep')
FDR_corrected_mainPendule_pos = mne.stats.fdr_correction(tableauValeurs_mainPendule, alpha=0.05, method='poscorr')


from numpy import genfromtxt
cortexMoteur_mainPendule = genfromtxt("../excel_files/avec_BL/t_test_23sujets_elecFreq_mainPendule_cortexMoteur.csv", delimiter=',')


FDR_corrected_mainPendule_cortexMoteur = mne.stats.fdr_correction(cortexMoteur_mainPendule, alpha=0.05, method='poscorr')#plutot method = poscorr

print(FDR_corrected_mainPendule_cortexMoteur)




#============== CALCULER la valeur moyenne =========================
for 




#=============== AFFICHER LES P VALUES =====================================
val_tTest_mainPendule_SMR = [0.1514285648,0.1948335038,0.6441319607,0.4198438029,0.6856312827,0.3678181708,0.380559897,0.6745968373,
0.6216189249,0.4481958297,0.4540524555,0.02161431175,0.7435749863,0.7408553258,0.17159543,
0.1426577926,0.2757693968,0.7027306385,0.5629363395,0.4519912642,0.2557785529,0.6262981274,0.8454071641,
0.3327393943,0.5157720749,0.545540985,0.4459905995,0.5658413497]

from math import log10
logValues = []
for val in val_tTest_mainPendule_SMR:
    logValues.append(20*log10(val))

import numpy as np
pvalues_array = np.zeros(shape=(28,1))
for i in range(len(val_tTest_mainPendule_SMR)):
    pvalues_array[i] = val_tTest_mainPendule_SMR[i]
    

ch_names = ['Fp1','Fp2','F7','F3','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','CP5',
'CP1','CP2','CP6','P7','P3','Pz','P4','P8','O1','Oz','O2','Fz']

info_array = mne.create_info(ch_names,ch_types='eeg',sfreq=1)
EvokedarrayPvalue = mne.EvokedArray(pvalues_array,info_array)

montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
EvokedarrayPvalue.set_montage(montageEasyCap)

import matplotlib.pyplot as plt
import numpy as np

my_cmap = discrete_log_cmap(14, 'Greens_r')

colors = [ "green","white"]
cmap, norm = cmap_logarithmic(0.001, 0.75, colors)
EvokedarrayPvalue.plot_topomap(times=0,vmin=0.01,cmap=cmap,units="pvalue",scalings=1)#, vmax=max(val_tTest_mainPendule_SMR)+0.05) 


cmap_list = [((0, 0.05), "green"), ((0.05, 0.1), "yellowgreen"), ((0.1, 1), "white")]
cmap_d, norm = cmap_discrete(cmap_list)
EvokedarrayPvalue.plot_topomap(times=0,vmin=0.01,vmax=1,cmap=cmap_d,units="pvalue",scalings=10)#, vmax=max(val_tTest_mainPendule_SMR)+0.05) 

raw_signal.plot(block=True)

# #============ MEME CHOSE SANS BASELINE, AVEC LES SEUILS : ON PERD LE PEU D'EFFET =====================
# ================= get the data===============
# from functions.load_savedData import *

# from handleData_subject import createSujetsData
# from functions.load_savedData import *
# from frequencyPower_displays import computeSlidingWindowEpochs

# essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates = createSujetsData()

# seuils_sujets = pd.read_csv("./data/seuil_data/seuils_sujets_dash.csv")

# #pour se placer dans les donnees lustre
# os.chdir("../../../../../../")
# lustre_data_dir = "iss02/cenir/analyse/meeg/BETAPARK/_RAW_DATA"
# lustre_path = pathlib.Path(lustre_data_dir)
# os.chdir(lustre_path)



# liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
# liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
# liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

# nbSujets = 24
# SujetsDejaTraites = 0
# rawPath_main_sujets = liste_rawPathMain[SujetsDejaTraites:SujetsDejaTraites+nbSujets]
# rawPath_pendule_sujets = liste_rawPathPendule[SujetsDejaTraites:SujetsDejaTraites+nbSujets]
# rawPath_mainIllusion_sujets = liste_rawPathMainIllusion[SujetsDejaTraites:SujetsDejaTraites+nbSujets]
# #================================= T -TEST AVEC SEUIL ==============================================
# #C3 8-30Hz, 23 sujets, 3 conditions pour commencer

# liste_tfrMain = load_tfr_data(rawPath_main_sujets,"")
# liste_tfrMainIllusion = load_tfr_data(rawPath_mainIllusion_sujets,"")
# liste_tfrPendule = load_tfr_data(rawPath_pendule_sujets,"")


# liste_tfr_mainIllusion = liste_tfrMainIllusion.copy()
# liste_tfr_main = liste_tfrMain.copy()
# liste_tfr_pendule = liste_tfrPendule.copy()
# # #APPLY SEUIL
# for i in range(23):
#     seuil = float(seuils_sujets["seuil_min_mvt"][i])
#     print(seuil)
#     main_seuil = liste_tfr_main[i].data/seuil
#     liste_tfr_main[i].data = main_seuil
#     pendule_seuil = liste_tfr_pendule[i].data/seuil
#     liste_tfr_pendule[i].data = pendule_seuil
#     mainIllusion_seuil = liste_tfr_mainIllusion[i].data/seuil
#     liste_tfr_mainIllusion[i].data = mainIllusion_seuil
    
# #crop time & frequency
# for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
#     tfr_mainI.crop(tmin = 1.5,tmax=25.5,fmin = 8,fmax = 30)
#     tfr_main.crop(tmin = 1.5,tmax=25.5,fmin = 8,fmax = 30)
#     tfr_pendule.crop(tmin = 1.5,tmax=25.5,fmin = 8,fmax = 30)
# #subset electrode
# for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
#     tfr_mainI.pick_channels(["C3"])
#     tfr_main.pick_channels(["C3"])
#     tfr_pendule.pick_channels(["C3"])
    
# #create t test table , with time and frequency
# tableauMain = np.zeros(shape=(23,24,23))#freqs x timepoints x sujets
# tableauMainIllusion = np.zeros(shape=(23,24,23))#freqs x timepoints x sujets
# tableauPendule = np.zeros(shape=(23,24,23))#freqs x timepoints x sujets
# for i in range(23):
#     print(i)
#     pendule_2dim = liste_tfr_pendule[i].data.mean(axis=0)#faire pour autres aussi
#     print(pendule_2dim.shape)
#     main_2dim = liste_tfr_main[i].data.mean(axis=0)#faire pour autres aussi
#     mainI_2dim = liste_tfrMainIllusion[i].data.mean(axis=0)#faire pour autres aussi
    
#     C3_mov_pendule = computeSlidingWindowEpochs(pendule_2dim,24)#a sauvegarder
#     C3_mov_main = computeSlidingWindowEpochs(main_2dim,24)
#     C3_mov_mainIllusion = computeSlidingWindowEpochs(mainI_2dim,24)


#     tableauPendule[:,:,i] = C3_mov_pendule
#     tableauMain[:,:,i] = C3_mov_main
#     tableauMainIllusion[:,:,i] = C3_mov_mainIllusion

# np.save('../numpy_files/C3_mov_pendule_8-30_23sujets_sansBL.npy', tableauPendule)
# np.save('../numpy_files/C3_mov_main_8-30_23sujets_sansBL.npy', tableauMain)
# np.save('../numpy_files/C3_mov_mainIllusion_8-30_23sujets_sansBL.npy', tableauMainIllusion)

# tableauPendule = np.load('../numpy_files/C3_mov_pendule_8-30_23sujets_sansBL.npy')
# tableauMain = np.load('../numpy_files/C3_mov_main_8-30_23sujets_sansBL.npy')
# tableauMainIllusion = np.load('../numpy_files/C3_mov_mainIllusion_8-30_23sujets_sansBL.npy')

# #save as excel
# writer = pd.ExcelWriter('../excel_files/C3_mov_pendule_8-30_23sujets_sansBL.xlsx', engine='xlsxwriter')
# for i in range(23):
#     df = pd.DataFrame(tableauPendule[:,:,i])
#     df.to_excel(writer, index=False,header=False,sheet_name='bin%d' % i)
# writer.save()
# writer = pd.ExcelWriter('../excel_files/C3_mov_main_8-30_23sujets_sansBL.xlsx', engine='xlsxwriter')
# for i in range(23):
#     df = pd.DataFrame(tableauMain[:,:,i])
#     df.to_excel(writer, index=False,header=False,sheet_name='bin%d' % i)
# writer.save()
# writer = pd.ExcelWriter('../excel_files/C3_mov_mainIllusion_8-30_23sujets_sansBL.xlsx', engine='xlsxwriter')
# for i in range(23):
#     df = pd.DataFrame(tableauMainIllusion[:,:,i])
#     df.to_excel(writer,index=False,header=False, sheet_name='bin%d' % i)
# writer.save()


# #=============get the data in the right format =========================
# tableauMain = pd.ExcelFile('../excel_files/C3_mov_main_8-30_23sujets_sansBL.xlsx')
# tableauPendule = pd.ExcelFile('../excel_files/C3_mov_pendule_8-30_23sujets_sansBL.xlsx')
# tableauMainIllusion = pd.ExcelFile('../excel_files/C3_mov_mainIllusion_8-30_23sujets_sansBL.xlsx')

# listeSujetsMain = []
# listeSujetsPendule = []
# listeSujetsMainIllusion = []

# for i in range(23):
#     df_Sujet_i_Main = pd.read_excel(tableauMain, "bin"+str(i),header=None)
#     listeSujetsMain.append(df_Sujet_i_Main)
    
#     df_Sujet_i_Pendule = pd.read_excel(tableauPendule, "bin"+str(i),header=None)
#     listeSujetsPendule.append(df_Sujet_i_Pendule)
    
#     df_Sujet_i_MainIllusion = pd.read_excel(tableauMainIllusion, "bin"+str(i),header=None)
#     listeSujetsMainIllusion.append(df_Sujet_i_MainIllusion)
    
    
# def get_t_testValue_cell(ligne,colonne):
#     print("ligne : "+str(ligne))
#     print("colonne : "+str(colonne))
#     ligne = ligne
#     colonne = colonne
#     listeValeurs_cellule_main = []
#     listeValeurs_cellule_pendule = []
#     listeValeurs_cellule_mainIllusion = []
    
#     for i in range(23):
#         listeValeurs_cellule_main.append(listeSujetsMain[i].iloc[ligne,colonne])
#         listeValeurs_cellule_pendule.append(listeSujetsPendule[i].iloc[ligne,colonne])
#         listeValeurs_cellule_mainIllusion.append(listeSujetsMainIllusion[i].iloc[ligne,colonne])

#     resultatMainPendule = scipy.stats.ttest_rel(listeValeurs_cellule_main,listeValeurs_cellule_pendule)
#     resultatMainMainIllusion = scipy.stats.ttest_rel(listeValeurs_cellule_main,listeValeurs_cellule_mainIllusion)

#     return resultatMainPendule,resultatMainMainIllusion


# #pour toutes les valeurs
# nbFreq = listeSujetsMain[0].shape[0] #lire le nb de points de frequence
# nbPointsTemps = listeSujetsMain[0].shape[1]-2#lire le nb de points de temps
# tableauValeurs_mainPendule = np.zeros(shape=(nbFreq,nbPointsTemps))
# tableauValeurs_mainMainIllusion = np.zeros(shape=(nbFreq,nbPointsTemps))
# for timePoint in range(nbPointsTemps):#colonne
#     for freqPoint in range(nbFreq):#ligne
#         res_mainPendule,res_mainMainI = get_t_testValue_cell(freqPoint,timePoint)
#         tableauValeurs_mainPendule[freqPoint,timePoint] = res_mainPendule[1]#p value
#         tableauValeurs_mainMainIllusion[freqPoint,timePoint] = res_mainMainI[1]#p value
        
    
# pd.DataFrame(tableauValeurs_mainPendule).to_csv("../excel_files/t_test_23sujets_tpsFreq_mainPendule_sansBL.csv",header=None, index=None)
# pd.DataFrame(tableauValeurs_mainMainIllusion).to_csv("../excel_files/t_test_23sujets_tpsFreq_mainMainI_sansBL.csv",header=None, index=None)


#==================== T-TEST ELECTRODE x FREQUENCE AVEC SEUIL ======================================
# from functions.load_savedData import *

# from handleData_subject import createSujetsData
# from functions.load_savedData import *
# from frequencyPower_displays import *

# essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates = createSujetsData()

# seuils_sujets = pd.read_csv("./data/seuil_data/seuils_sujets_dash.csv")

# #pour se placer dans les donnees lustre
# os.chdir("../../../../../../")
# lustre_data_dir = "iss02/cenir/analyse/meeg/BETAPARK/_RAW_DATA"
# lustre_path = pathlib.Path(lustre_data_dir)
# os.chdir(lustre_path)



# liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
# liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
# liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

# nbSujets = 24
# SujetsDejaTraites = 0
# rawPath_main_sujets = liste_rawPathMain[SujetsDejaTraites:SujetsDejaTraites+nbSujets]
# rawPath_pendule_sujets = liste_rawPathPendule[SujetsDejaTraites:SujetsDejaTraites+nbSujets]
# rawPath_mainIllusion_sujets = liste_rawPathMainIllusion[SujetsDejaTraites:SujetsDejaTraites+nbSujets]

# liste_tfrMain = load_tfr_data(rawPath_main_sujets,"")
# liste_tfrMainIllusion = load_tfr_data(rawPath_mainIllusion_sujets,"")
# liste_tfrPendule = load_tfr_data(rawPath_pendule_sujets,"")


# liste_tfr_mainIllusion = liste_tfrMainIllusion.copy()
# liste_tfr_main = liste_tfrMain.copy()
# liste_tfr_pendule = liste_tfrPendule.copy()

# # #APPLY SEUIL
# for i in range(23):
#     seuil = float(seuils_sujets["seuil_min_mvt"][i])
#     print(seuil)
#     main_seuil = liste_tfr_main[i].data/seuil
#     liste_tfr_main[i].data = main_seuil
#     pendule_seuil = liste_tfr_pendule[i].data/seuil
#     liste_tfr_pendule[i].data = pendule_seuil
#     mainIllusion_seuil = liste_tfr_mainIllusion[i].data/seuil
#     liste_tfr_mainIllusion[i].data = mainIllusion_seuil
    

# #crop time & frequency
# for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
#     tfr_mainI.crop(tmin = 1.5,tmax=25.5,fmin = 8,fmax = 30)
#     tfr_main.crop(tmin = 1.5,tmax=25.5,fmin = 8,fmax = 30)
#     tfr_pendule.crop(tmin = 1.5,tmax=25.5,fmin = 8,fmax = 30)


# #create t test table , with electrodes and frequency
# tableauMain = np.zeros(shape=(28,23,23))#electrodes x freqs x sujets
# tableauMainIllusion = np.zeros(shape=(28,23,23))#electrodes x freqs x sujets
# tableauPendule = np.zeros(shape=(28,23,23))#electrodes x freqs x sujets
# for i in range(23):
#     allElec_mov_pendule = computeSlidingWindowEpochs_elecFreq(liste_tfr_pendule[i].data,24)
#     allElec_mov_main = computeSlidingWindowEpochs_elecFreq(liste_tfr_main[i].data,24)
#     allElec_mov_mainIllusion = computeSlidingWindowEpochs_elecFreq(liste_tfr_mainIllusion[i].data,24)
#     #moyenner les 24 points de temps
#     allElec_mov_pendule_timePooled = allElec_mov_pendule.mean(axis=2)
#     allElec_mov_main_timePooled = allElec_mov_main.mean(axis=2)
#     allElec_mov_mainIllusion_timePooled = allElec_mov_mainIllusion.mean(axis=2)
    

#     tableauPendule[:,:,i] = allElec_mov_pendule_timePooled
#     tableauMain[:,:,i] = allElec_mov_main_timePooled
#     tableauMainIllusion[:,:,i] = allElec_mov_mainIllusion_timePooled

# np.save('../numpy_files/allElec_mov_pendule_timePooled_8-30_23sujets.npy', tableauPendule)
# np.save('../numpy_files/allElec_mov_main_timePooled_8-30_23sujets.npy', tableauMain)
# np.save('../numpy_files/allElec_mov_mainIllusion_timePooled_8-30_23sujets.npy', tableauMainIllusion)

# tableauPendule = np.load('../numpy_files/allElec_mov_pendule_timePooled_8-30_23sujets.npy')
# tableauMain = np.load('../numpy_files/allElec_mov_main_timePooled_8-30_23sujets.npy')
# tableauMainIllusion = np.load('../numpy_files/allElec_mov_mainIllusion_timePooled_8-30_23sujets.npy')

# #save as excel
# writer = pd.ExcelWriter('../excel_files/sans_BL/All_elecs/allElec_mov_pendule_timePooled_8-30_23sujets.xlsx', engine='xlsxwriter')
# for i in range(23):
#     df = pd.DataFrame(tableauPendule[:,:,i])
#     df.to_excel(writer,index=False,header=False, sheet_name='bin%d' % i)
# writer.save()
# writer = pd.ExcelWriter('../excel_files/sans_BL/All_elecs/allElec_mov_main_timePooled_8-30_23sujets.xlsx', engine='xlsxwriter')
# for i in range(23):
#     df = pd.DataFrame(tableauMain[:,:,i])
#     df.to_excel(writer,index=False,header=False, sheet_name='bin%d' % i)
# writer.save()
# writer = pd.ExcelWriter('../excel_files/sans_BL/All_elecs/allElec_mov_mainIllusion_timePooled_8-30_23sujets.xlsx', engine='xlsxwriter')
# for i in range(23):
#     df = pd.DataFrame(tableauMainIllusion[:,:,i])
#     df.to_excel(writer,index=False,header=False, sheet_name='bin%d' % i)
# writer.save()


# #=============get the data in the right format =========================
# tableauMain = pd.ExcelFile('../excel_files/sans_BL/All_elecs/allElec_mov_main_timePooled_8-30_23sujets.xlsx')
# tableauPendule = pd.ExcelFile('../excel_files/sans_BL/All_elecs/allElec_mov_pendule_timePooled_8-30_23sujets.xlsx')
# tableauMainIllusion = pd.ExcelFile('../excel_files/sans_BL/All_elecs/allElec_mov_mainIllusion_timePooled_8-30_23sujets.xlsx')

# listeSujetsMain = []
# listeSujetsPendule = []
# listeSujetsMainIllusion = []

# for i in range(23):
#     df_Sujet_i_Main = pd.read_excel(tableauMain, "bin"+str(i),header=None)#,index=False)
#     listeSujetsMain.append(df_Sujet_i_Main)
    
#     df_Sujet_i_Pendule = pd.read_excel(tableauPendule, "bin"+str(i),header=None)
#     listeSujetsPendule.append(df_Sujet_i_Pendule)
    
#     df_Sujet_i_MainIllusion = pd.read_excel(tableauMainIllusion, "bin"+str(i),header=None)
#     listeSujetsMainIllusion.append(df_Sujet_i_MainIllusion)
    
    
# def get_t_testValue_cell(ligne,colonne):
#     print("ligne : "+str(ligne))
#     print("colonne : "+str(colonne))
#     ligne = ligne
#     colonne = colonne
#     listeValeurs_cellule_main = []
#     listeValeurs_cellule_pendule = []
#     listeValeurs_cellule_mainIllusion = []
    
#     for i in range(23):
#         listeValeurs_cellule_main.append(listeSujetsMain[i].iloc[ligne,colonne])#l'indexage est base sur les noms de colonne
#         listeValeurs_cellule_pendule.append(listeSujetsPendule[i].iloc[ligne,colonne])#, ne marche plus quand on vire le header
#         listeValeurs_cellule_mainIllusion.append(listeSujetsMainIllusion[i].iloc[ligne,colonne])
#     resultatMainPendule = scipy.stats.ttest_rel(listeValeurs_cellule_main,listeValeurs_cellule_pendule)
#     resultatMainMainIllusion = scipy.stats.ttest_rel(listeValeurs_cellule_main,listeValeurs_cellule_mainIllusion)

#     return resultatMainPendule,resultatMainMainIllusion


# #pour toutes les valeurs
# nbElec = listeSujetsMain[0].shape[0] #lire le nb d'electrodes'
# nbFreq = listeSujetsMain[0].shape[1]#lire le nb de frequences
# tableauValeurs_mainPendule = np.zeros(shape=(nbElec,nbFreq))
# tableauValeurs_mainMainIllusion = np.zeros(shape=(nbElec,nbFreq))
# for freqPoint in range(nbFreq):#colonne
#     for elec in range(nbElec):#ligne
#         print(elec)
#         res_mainPendule,res_mainMainI = get_t_testValue_cell(elec,freqPoint)
#         tableauValeurs_mainPendule[elec,freqPoint] = res_mainPendule[1]#p value
#         tableauValeurs_mainMainIllusion[elec,freqPoint] = res_mainMainI[1]#p value
        
    
# pd.DataFrame(tableauValeurs_mainPendule).to_csv("../excel_files/sans_BL/All_elecs/t_test_23sujets_elecFreq_mainPendule.csv",header=None, index=None)
# pd.DataFrame(tableauValeurs_mainMainIllusion).to_csv("../excel_files/sans_BL/All_elecs/t_test_23sujets_elecFreq_mainMainI.csv",header=None, index=None)








