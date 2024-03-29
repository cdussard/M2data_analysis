#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:44:43 2022

@author: claire.dussard
"""
#av_power_main_noBL_seuil,av_power_mainIllusion_noBL_seuil,av_power_pendule_noBL_seuil

liste_tfrMain = load_tfr_data(rawPath_main_sujets,"")
liste_tfr_main = liste_tfrMain.copy()
liste_tfrMainIllusion = load_tfr_data(rawPath_mainIllusion_sujets,"")
liste_tfr_mainIllusion = liste_tfrMainIllusion.copy()
liste_tfrPendule = load_tfr_data(rawPath_pendule_sujets,"")
liste_tfr_pendule = liste_tfrPendule.copy()
#compute seuil au lieu de baseline
for i in range(23):
    seuil = float(seuils_sujets["seuil_min_mvt"][i])
    print(seuil)
    main_seuil = liste_tfr_main[i].data/seuil
    liste_tfr_main[i].data = main_seuil
    pendule_seuil = liste_tfr_pendule[i].data/seuil
    liste_tfr_pendule[i].data = pendule_seuil
    mainIllusion_seuil = liste_tfr_mainIllusion[i].data/seuil
    liste_tfr_mainIllusion[i].data = mainIllusion_seuil

#crop time & frequency
for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
    tfr_mainI.crop(tmin = 2,tmax=26,fmin = 8,fmax = 30)
    tfr_main.crop(tmin = 2,tmax=26,fmin = 8,fmax = 30)
    tfr_pendule.crop(tmin = 2,tmax=26,fmin = 8,fmax = 30)
#subset electrode
for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
    tfr_mainI.pick_channels(["C3"])
    tfr_main.pick_channels(["C3"])
    tfr_pendule.pick_channels(["C3"])
#create ANOVA table
tableauANOVA = np.zeros(shape=(23,3))
tableauANOVAmediane = np.zeros(shape=(23,3))
for i in range(23):
    #pool power
    powerOverTime8_30Hz_pendule = np.mean(liste_tfr_pendule[i].data,axis=1)
    powerOverTime8_30Hz_main = np.mean(liste_tfr_main[i].data,axis=1)
    powerOverTime8_30Hz_mainI = np.mean(liste_tfr_mainIllusion[i].data,axis=1)
    #pool time
    valuePower8_30Hz_pendule = np.mean(powerOverTime8_30Hz_pendule.data,axis=1)[0] #pour le dernier sujet
    valuePower8_30Hz_main = np.mean(powerOverTime8_30Hz_main.data,axis=1)[0]
    valuePower8_30Hz_mainIllusion = np.mean(powerOverTime8_30Hz_mainI.data,axis=1)[0]
    valuePower8_30Hz_pendule_med = np.median(powerOverTime8_30Hz_pendule.data,axis=1)[0] #pour le dernier sujet
    valuePower8_30Hz_main_med = np.median(powerOverTime8_30Hz_main.data,axis=1)[0]
    valuePower8_30Hz_mainIllusion_med = np.median(powerOverTime8_30Hz_mainI.data,axis=1)[0]
    tableauANOVA[i][0] = valuePower8_30Hz_pendule
    tableauANOVA[i][1] = valuePower8_30Hz_main
    tableauANOVA[i][2] = valuePower8_30Hz_mainIllusion
    tableauANOVAmediane[i][0] = valuePower8_30Hz_pendule_med
    tableauANOVAmediane[i][1] = valuePower8_30Hz_main_med
    tableauANOVAmediane[i][2] = valuePower8_30Hz_mainIllusion_med

tableauANOVA_NoBL_seuil_mean = tableauANOVA
pd.DataFrame(tableauANOVA_NoBL_seuil_mean).to_csv("../csv_files/ANOVA_C3_noBL_seuil/tableauANOVA_mean_Seuil_C3_8-30Hz.csv")

tableauANOVA_NoBL_seuil_med = tableauANOVAmediane
pd.DataFrame(tableauANOVA_NoBL_seuil_med).to_csv("../csv_files/ANOVA_C3_noBL_seuil/tableauANOVA_med_Seuil_C3_8-30Hz.csv")

#=====================================same en laplacien ==============

liste_power_sujets_main_C3Lap = load_tfr_data(rawPath_main_sujets,"C3C4laplacien")

liste_power_sujets_pendule_C3Lap = load_tfr_data(rawPath_pendule_sujets,"C3C4laplacien") 

liste_power_sujets_mainIllusion_C3Lap = load_tfr_data(rawPath_mainIllusion_sujets,"C3C4laplacien")

liste_tfr_main = liste_power_sujets_main_C3Lap
liste_tfr_pendule = liste_power_sujets_pendule_C3Lap
liste_tfr_mainIllusion = liste_power_sujets_mainIllusion_C3Lap


#crop time & frequency
for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
    tfr_mainI.crop(tmin = 2,tmax=26,fmin = 8,fmax = 30)
    tfr_main.crop(tmin = 2,tmax=26,fmin = 8,fmax = 30)
    tfr_pendule.crop(tmin = 2,tmax=26,fmin = 8,fmax = 30)
#subset electrode
for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
    tfr_mainI.pick_channels(["C3"])
    tfr_main.pick_channels(["C3"])
    tfr_pendule.pick_channels(["C3"])
#create ANOVA table
tableauANOVA = np.zeros(shape=(23,3))
tableauANOVAmediane = np.zeros(shape=(23,3))
for i in range(23):
    #pool power
    powerOverTime8_30Hz_pendule = np.mean(liste_tfr_pendule[i].data,axis=1)
    powerOverTime8_30Hz_main = np.mean(liste_tfr_main[i].data,axis=1)
    powerOverTime8_30Hz_mainI = np.mean(liste_tfr_mainIllusion[i].data,axis=1)
    print(powerOverTime8_30Hz_mainI)
    print(powerOverTime8_30Hz_mainI.data)
    #pool time
    valuePower8_30Hz_pendule = np.mean(powerOverTime8_30Hz_pendule.data,axis=1)[0] #pour le dernier sujet
    valuePower8_30Hz_main = np.mean(powerOverTime8_30Hz_main.data,axis=1)[0]
    valuePower8_30Hz_mainIllusion = np.mean(powerOverTime8_30Hz_mainI.data,axis=1)[0]
    valuePower8_30Hz_pendule_med = np.median(powerOverTime8_30Hz_pendule.data,axis=1)[0] #pour le dernier sujet
    valuePower8_30Hz_main_med = np.median(powerOverTime8_30Hz_main.data,axis=1)[0]
    valuePower8_30Hz_mainIllusion_med = np.median(powerOverTime8_30Hz_mainI.data,axis=1)[0]
    tableauANOVA[i][0] = valuePower8_30Hz_pendule
    tableauANOVA[i][1] = valuePower8_30Hz_main
    tableauANOVA[i][2] = valuePower8_30Hz_mainIllusion
    tableauANOVAmediane[i][0] = valuePower8_30Hz_pendule_med
    tableauANOVAmediane[i][1] = valuePower8_30Hz_main_med
    tableauANOVAmediane[i][2] = valuePower8_30Hz_mainIllusion_med

tableauANOVA_NoBL_seuil_meanC3Lapl = tableauANOVA
pd.DataFrame(tableauANOVA_NoBL_seuil_meanC3Lapl).to_csv("../csv_files/ANOVA_C3_noBL_seuil/tableauANOVA_mean_Seuil_C3Lapl_8-30Hz.csv")

tableauANOVA_NoBL_seuil_medC3Lapl = tableauANOVAmediane
pd.DataFrame(tableauANOVA_NoBL_seuil_medC3Lapl).to_csv("../csv_files/ANOVA_C3_noBL_seuil/tableauANOVA_med_Seuil_C3Lapl_8-30Hz.csv")

# #=========== decoupage openvibe ================
# def computeMovingAverage_openvibe(C3values,nvalues):#35 pour tout data, 24 si crop #FONCTION A REVOIR
#     arr_C3_movAverage = list()
#     compteur_moyenne = 1
#     for i in range(1,nvalues):
#         print("n value"+str(i))
#         if compteur_moyenne == 5:
#             print("continue")
#             compteur_moyenne += 1
#             continue#passe l'instance de la boucle
#         elif compteur_moyenne == 6:
#             compteur_moyenne = 1
#             print("continue")
#             continue
#         offset = 125*i
#         point_1 = C3values[:,250*i :250*(i+1) ].mean()
#         point_2 = C3values[:,63+(250*i):63 + (250*(i+1))].mean()
#         point_3 = C3values[:,125+(250*i) :125 +(250*(i+1))].mean()
#         point_4 = C3values[:,188+(250*i) :188 +(250*(i+1)) ].mean()
#         pointMoyenne = (point_1+point_2+point_3 + point_4)/4
#         print(pointMoyenne)
#         arr_C3_movAverage.append(pointMoyenne)
#         compteur_moyenne += 1
#     print(len(arr_C3_movAverage))
#     return arr_C3_movAverage


liste_tfr_main = liste_power_sujets_main_C3Lap
liste_tfr_pendule = liste_power_sujets_pendule_C3Lap
liste_tfr_mainIllusion = liste_power_sujets_mainIllusion_C3Lap


#crop time & frequency
for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
    tfr_mainI.crop(tmin = 2,tmax=26,fmin = 8,fmax = 30)
    tfr_main.crop(tmin = 2,tmax=26,fmin = 8,fmax = 30)
    tfr_pendule.crop(tmin = 2,tmax=26,fmin = 8,fmax = 30)
#subset electrode
for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
    tfr_mainI.pick_channels(["C3"])
    tfr_main.pick_channels(["C3"])
    tfr_pendule.pick_channels(["C3"])
#create ANOVA table
tableauANOVA = np.zeros(shape=(23,3))
tableauANOVAmediane = np.zeros(shape=(23,3))
for i in range(23):
    #pool power
    powerOverTime8_30Hz_pendule = np.mean(liste_tfr_pendule[i].data,axis=1)
    powerOverTime8_30Hz_main = np.mean(liste_tfr_main[i].data,axis=1)
    powerOverTime8_30Hz_mainI = np.mean(liste_tfr_mainIllusion[i].data,axis=1)
    #pool time
    listPendule = computeMovingAverage_openvibe(powerOverTime8_30Hz_pendule,23)
    listMain = computeMovingAverage_openvibe(powerOverTime8_30Hz_main,23)
    listMainIllusion = computeMovingAverage_openvibe(powerOverTime8_30Hz_mainI,23)
    
    valuePower8_30Hz_pendule = np.mean(listPendule)
    valuePower8_30Hz_main = np.mean(listMain)
    valuePower8_30Hz_mainIllusion = np.mean(listMainIllusion)
    valuePower8_30Hz_pendule_med = np.median(listPendule)
    valuePower8_30Hz_main_med = np.median(listMain)
    valuePower8_30Hz_mainIllusion_med = np.median(listMainIllusion)
    tableauANOVA[i][0] = valuePower8_30Hz_pendule
    tableauANOVA[i][1] = valuePower8_30Hz_main
    tableauANOVA[i][2] = valuePower8_30Hz_mainIllusion
    tableauANOVAmediane[i][0] = valuePower8_30Hz_pendule_med
    tableauANOVAmediane[i][1] = valuePower8_30Hz_main_med
    tableauANOVAmediane[i][2] = valuePower8_30Hz_mainIllusion_med

tableauANOVA_NoBL_seuil_meanC3Lapl_overlap = tableauANOVA
pd.DataFrame(tableauANOVA_NoBL_seuil_meanC3Lapl_overlap).to_csv("../csv_files/ANOVA_C3_noBL_seuil/tableauANOVA_mean_Seuil_C3Lapl_overlap_8-30Hz.csv")

tableauANOVA_NoBL_seuil_medC3Lapl_overlap = tableauANOVAmediane
pd.DataFrame(tableauANOVA_NoBL_seuil_medC3Lapl_overlap).to_csv("../csv_files/ANOVA_C3_noBL_seuil/tableauANOVA_med_Seuil_C3Lapl_overlap_8-30Hz.csv")

#