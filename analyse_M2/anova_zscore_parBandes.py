# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:48:40 2022

@author: claire.dussard
"""
from functions.load_savedData import *
from handleData_subject import createSujetsData
from functions.load_savedData import *
from functions.frequencyPower_displays import *
import numpy as np
import os
import pandas as pd

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()

#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

liste_tfrPendule = load_tfr_data_windows(liste_rawPathPendule,"",True)
liste_tfrMain = load_tfr_data_windows(liste_rawPathMain,"",True)
liste_tfrMainIllusion = load_tfr_data_windows(liste_rawPathMainIllusion,"",True)


def copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion):
#avoid having to reload from scratch after every ANOVA (instances modified in place)
    liste_tfr_pendule = []
    liste_tfr_main = []
    liste_tfr_mainIllusion = []
    for tfr_p,tfr_m,tfr_mi in zip(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion):
        liste_tfr_pendule.append(tfr_p.copy())
        liste_tfr_main.append(tfr_m.copy())
        liste_tfr_mainIllusion.append(tfr_mi.copy())
    return liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion




def anova_data_elec(liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule,mode_baseline,fmin,fmax,tmin,tmax,elec):
    baseline = (-3,-1)
    #compute baseline (first because after we crop time)
    for tfr_m,tfr_mi,tfr_p in zip(liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule):
        tfr_m.apply_baseline(baseline=baseline, mode=mode_baseline, verbose=None)
        tfr_mi.apply_baseline(baseline=baseline, mode=mode_baseline, verbose=None)
        tfr_p.apply_baseline(baseline=baseline, mode=mode_baseline, verbose=None)
    #crop time & frequency
    for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
        tfr_mainI.crop(tmin = tmin,tmax=tmax,fmin = fmin,fmax = fmax)
        tfr_main.crop(tmin = tmin,tmax=tmax,fmin = fmin,fmax = fmax)
        tfr_pendule.crop(tmin = tmin,tmax=tmax,fmin = fmin,fmax = fmax)
    #subset electrode
    for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
        tfr_mainI.pick_channels([elec])
        tfr_main.pick_channels([elec])
        tfr_pendule.pick_channels([elec])
  
    #create ANOVA table "faire evoluer pour plusieurs elecs a la fois
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
    return tableauANOVA,tableauANOVAmediane


#ANOVA
from statsmodels.stats.anova import AnovaRM
def ANOVA_result(tableauANOVA_moyenne,tableauANOVA_mediane,save,nameSave):
    
    allERDmediane = [val for val in zip(tableauANOVA_mediane[:,0],tableauANOVA_mediane[:,1],tableauANOVA_mediane[:,2])]
    allERDmean = [val for val in zip(tableauANOVA_moyenne[:,0],tableauANOVA_moyenne[:,1],tableauANOVA_moyenne[:,2])]
    
    allERDmediane = list(sum(allERDmediane,()))
    allERDmean = list(sum(allERDmean,()))
    
    df_mean = pd.DataFrame({'condition': np.tile(["pendule", "main","mainIllusion"],23),#pendule main mainIllusion
                       'sujet': np.repeat([0,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],3),
                          'ERD':allERDmean})
    df_mediane = pd.DataFrame({'condition': np.tile(["pendule", "main","mainIllusion"],23),
                       'sujet': np.repeat([0,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],3),
                          'ERD':allERDmediane})
    # df_2cond_mediane = df_mediane[df_mediane["condition"]!=3]
    # df_2cond_mediane_main = df_mediane[df_mediane["condition"]!=1]
    
    anovaMediane = AnovaRM(data=df_mediane, depvar='ERD', subject='sujet', within=['condition']).fit()
    anovaMean = AnovaRM(data=df_mean, depvar='ERD', subject='sujet', within=['condition']).fit()
    print(anovaMediane)
    print(anovaMean)
    return anovaMediane,anovaMean,df_mean,df_mediane


#def t_test_result(tableauANOVA_moyenne,tableauANOVA_mediane)

#faire les t tests post hoc main vs pendule 8-30 Hz + pour bandes signif
#faire les corrections en FDR par ex sur 8-12Hz / 12/15Hz / 13-20Hz / 20-30Hz


bandsToTest = [[8,12],[12,15],[13,20],[20,30],[13,35]]
electrodesToTest = ["C3","C4","Cz","FC1","FC5","CP1","CP5"]
Ftable_tableau_moyenne = np.zeros(shape=(len(bandsToTest),len(electrodesToTest)))
pvalueTable_tableau_moyenne = np.zeros(shape=(len(bandsToTest),len(electrodesToTest)))
Ftable_tableau_mediane = np.zeros(shape=(len(bandsToTest),len(electrodesToTest)))
pvalueTable_tableau_mediane = np.zeros(shape=(len(bandsToTest),len(electrodesToTest)))
i = 0
j = 0
for band in bandsToTest:
    print(str(band))
    for elec in electrodesToTest:
        print(str(elec)) 
        #copy
        liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion = copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion)
        #filtrer et moyenner
        moy_bandElec,med_bandElec = anova_data_elec(liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule,"logratio",band[0],band[1],2,25,elec)
        anovaMediane,anovaMean = ANOVA_result(moy_bandElec,med_bandElec,False,"")
        anovaMediane_table = anovaMediane.anova_table.values[0]
        anovaMean_table = anovaMean.anova_table.values[0]
        print("pvalue mean for "+elec+" "+str(band[0])+"-"+str(band[1])+"Hz : "+str(round(anovaMean_table[3],4)))
        pvalueTable_tableau_moyenne[i,j] = anovaMean_table[3]
        Ftable_tableau_moyenne [i,j ] = anovaMean_table[0]
        print("pvalue median for "+elec+" "+str(band[0])+"-"+str(band[1])+"Hz : "+str(round(anovaMediane_table[3],4)))
        pvalueTable_tableau_mediane [i,j ] = anovaMediane_table[3]
        Ftable_tableau_mediane [i,j ] = anovaMediane_table[0]
        j+= 1 
    j = 0
    i+= 1

liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion = copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion)
moy_bandElec_c3_12_15hz,med_bandElec_c3_12_15hz= anova_data_elec(liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule,"logratio",12,15,2.5,26.8,"C3")
anovaMediane_c3_12_15hz,anovaMean_c3_12_15hz,df_mean_12_15hz,df_mediane_12_15hz = ANOVA_result(moy_bandElec_c3_12_15hz,med_bandElec_c3_12_15hz,False,"")

#1 ANOVA POUR 4 BANDES
liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion = copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion)
moy_bandElec_c3_8_12hz,med_bandElec_c3_8_12hz= anova_data_elec(liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule,"logratio",8,12,2.5,26.8,"C3")
liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion = copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion)
moy_bandElec_c3_12_15hz,med_bandElec_c3_12_15hz= anova_data_elec(liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule,"logratio",12,15,2.5,26.8,"C3")
liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion = copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion)
moy_bandElec_c3_13_20hz,med_bandElec_c3_13_20hz= anova_data_elec(liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule,"logratio",13,20,2.5,26.8,"C3")
liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion = copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion)
moy_bandElec_c3_15_20hz,med_bandElec_c3_15_20hz= anova_data_elec(liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule,"logratio",15,20,2.5,26.8,"C3")
liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion = copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion)
moy_bandElec_c3_20_30hz,med_bandElec_c3_20_30hz= anova_data_elec(liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule,"logratio",20,30,2.5,26.8,"C3")
liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion = copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion)
moy_bandElec_c3_8_13hz,med_bandElec_c3_8_13hz= anova_data_elec(liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule,"logratio",8,13,2.5,26.8,"C3")
liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion = copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion)
moy_bandElec_c3_15_30hz,med_bandElec_c3_15_30hz= anova_data_elec(liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule,"logratio",15,30,2.5,26.8,"C3")



listeRes = [med_bandElec_c3_8_12hz,med_bandElec_c3_12_15hz,med_bandElec_c3_13_20hz,med_bandElec_c3_15_20hz,med_bandElec_c3_20_30hz]
listeString = ["8_12Hz","12_15Hz","13_20Hz","15_20Hz","20_30Hz"]
for res,txt in zip(listeRes,listeString):
    df_long = pd.DataFrame(res)
    df_long.to_csv("../csv_files/ANOVA_bandes/ANOVA_"+txt+"_C3_long.csv")

df_long = pd.DataFrame(med_bandElec_c3_8_13hz)
df_long.to_csv("../csv_files/ANOVA_bandes/ANOVA_8_13Hz_C3_long.csv")

df_long = pd.DataFrame(med_bandElec_c3_15_30hz)
df_long.to_csv("../csv_files/ANOVA_bandes/ANOVA_15_30Hz_C3_long.csv")

import ptitprince
#raincloud plot
plt.figure()
dfData = df_mediane_12_15hz
douzeQuinzeHzData = pd.read_csv("./data/Jasp_anova/ANOVA_12_15Hz_C3_long.csv")
dfData.to_csv("./data/Jasp_anova/ANOVA_12_15Hz_C3_short_med.csv")
#8dfData = pd.read_csv("./data/Jasp_anova/ANOVA_12_15Hz_C3_short_med.csv")
df_long = pd.DataFrame(med_bandElec_c3_12_15hz)
df_long.to_csv("./data/Jasp_anova/ANOVA_12_15Hz_C3_long.csv")
ptitprince.RainCloud(data = df_mediane_12_15hz, x = 'condition', y = 'ERD', orient = 'v',pointplot = True)
plt.figure()
ptitprince.RainCloud(data = df_mean_12_15hz, x = 'condition', y = 'ERD', orient = 'v',pointplot = True)
raw_signal.plot(block=True)

#t test sur les valeurs
import scipy
scipy.stats.ttest_rel(moy_bandElec_c3_12_15hz[:,0],moy_bandElec_c3_12_15hz[:,1])#main vs pendule
scipy.stats.ttest_rel(moy_bandElec_c3_12_15hz[:,1],moy_bandElec_c3_12_15hz[:,2])#main vs mainIllusion

#8-30Hz
liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion = copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion)
moy_bandElec_c3_8_30hz,med_bandElec_c3_8_30hz = anova_data_elec(liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule,"logratio",8,30,2.5,26.8,"C3")
anovaMediane_c3_8_30hz,anovaMean_c3_8_30hz,df_mean_8_30hz,df_mediane_8_30hz = ANOVA_result(moy_bandElec_c3_8_30hz,med_bandElec_c3_8_30hz,False,"")
dfData = df_mediane_8_30hz
dfData.to_csv("./data/Jasp_anova/ANOVA_8_30Hz_C3_short_med.csv")
df_long = pd.DataFrame(med_bandElec_c3_8_30hz)
df_long.to_csv("./data/Jasp_anova/ANOVA_8_30Hz_C3_long.csv")
# df_long = pd.DataFrame(moy_bandElec_c3_8_30hz)
# df_long.to_csv("./data/Jasp_anova/ANOVA_8_30Hz_C3_long.csv")
# ptitprince.RainCloud(data = dfData, x = 'condition', y = 'ERD', orient = 'v')
# raw_signal.plot(block=True)

#post hoc 8-30Hz
scipy.stats.ttest_rel(moy_bandElec_c3_8_30hz[:,0],moy_bandElec_c3_8_30hz[:,1])#main vs pendule
scipy.stats.ttest_rel(moy_bandElec_c3_8_30hz[:,1],moy_bandElec_c3_8_30hz[:,2])#main vs mainIllusion
# tableauANOVA_NoBL_seuil_mean = tableauANOVA
# pd.DataFrame(tableauANOVA_NoBL_seuil_mean).to_csv("../csv_files/ANOVA_C3_noBL_seuil/tableauANOVA_mean_Seuil_C3_8-30Hz.csv")
_
# tableauANOVA_NoBL_seuil_med = tableauANOVAmediane
# pd.DataFrame(tableauANOVA_NoBL_seuil_med).to_csv("../csv_files/ANOVA_C3_noBL_seuil/tableauANOVA_med_Seuil_C3_8-30Hz.csv")

#12-15Hz CP5, CP1, Fc1, FC5
liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion = copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion)
moy_bandElec_CP5_12_15hz,med_bandElec_CP5_12_15hz = anova_data_elec(liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule,"logratio",12,15,2.5,26.8,"CP5")
anovaMediane_CP5_12_15hz,anovaMean_CP5_12_15hz,df_mean_12_15hz,df_mediane_12_15hz = ANOVA_result(moy_bandElec_CP5_12_15hz,med_bandElec_CP5_12_15hz,False,"")
#0.15;0.12

liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion = copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion)
moy_bandElec_CP1_12_15hz,med_bandElec_CP1_12_15hz = anova_data_elec(liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule,"logratio",12,15,2.5,26.8,"CP1")
anovaMediane_CP1_12_15hz,anovaMean_CP1_12_15hz,df_mean_12_15hz,df_mediane_12_15hz = ANOVA_result(moy_bandElec_CP1_12_15hz,med_bandElec_CP1_12_15hz,False,"")
#0.38 / 0.4

liste_tfr_pendule,liste_tfr_main,liste_tfr_mainIllusion = copy_three_tfrs(liste_tfrPendule,liste_tfrMain,liste_tfrMainIllusion)
moy_bandElec_CP1_12_15hz,med_bandElec_CP1_12_15hz = anova_data_elec(liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule,"logratio",12,15,2.5,26.8,"CP1")
anovaMediane_CP1_12_15hz,anovaMean_CP1_12_15hz,df_mean_12_15hz,df_mediane_12_15hz = ANOVA_result(moy_bandElec_CP1_12_15hz,med_bandElec_CP1_12_15hz,False,"")

