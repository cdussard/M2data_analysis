# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:15:44 2023

@author: claire.dussard
"""

import os 
import seaborn as sns
import pathlib
import mne
import pandas as pd
#necessite d'avoir execute handleData_subject.py, et load_savedData avant 
import numpy as np 
# importer les fonctions definies par moi 
from handleData_subject import createSujetsData
from functions.load_savedData import *
from functions.preprocessData_eogRefait import *
essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()


liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)
 

liste_tfrMain = load_tfr_data_windows(liste_rawPathMain,"",True)
liste_tfrMainIllusion = load_tfr_data_windows(liste_rawPathMainIllusion,"",True)
liste_tfrPendule = load_tfr_data_windows(liste_rawPathPendule,"",True)

fmin = 17
fmax = 25
elec = "C3"

liste_tfr_main = liste_tfrMain.copy()
liste_tfr_mainIllusion = liste_tfrMainIllusion.copy()
liste_tfr_pendule = liste_tfrPendule.copy()


#compute baseline
baseline = (-3,-1)
for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
    tfr_mainI.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    tfr_pendule.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    tfr_main.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
#crop time & frequency
for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
    tfr_mainI.crop(tmin = 2.5,tmax=26.8,fmin = fmin,fmax = fmax)
    tfr_main.crop(tmin = 2.5,tmax=26.8,fmin = fmin,fmax = fmax)
    tfr_pendule.crop(tmin = 2.5,tmax=26.8,fmin = fmin,fmax = fmax)
#subset electrode
for tfr_mainI,tfr_main,tfr_pendule in zip(liste_tfr_mainIllusion,liste_tfr_main,liste_tfr_pendule):
    tfr_mainI.pick_channels([elec])
    tfr_main.pick_channels([elec])
    tfr_pendule.pick_channels([elec])
#create ANOVA table
n_suj = 23
tableauANOVA = np.zeros(shape=(n_suj,3))
tableauANOVAmediane = np.zeros(shape=(n_suj,3))
for i in range(n_suj):
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

pd.DataFrame(tableauANOVA).to_csv("../csv_files/H3_anova/"+elec+"_"+str(fmin)+"_"+str(fmax)+"Hz_redone_anova.csv")

pd.DataFrame(tableauANOVAmediane).to_csv("../csv_files/H3_anova/"+elec+"_"+str(fmin)+"_"+str(fmax)+"Hz_redone_anova_mediane.csv")


from statsmodels.stats.anova import AnovaRM
allERDmediane = [val for val in zip(tableauANOVAmediane[:,0],tableauANOVAmediane[:,1],tableauANOVAmediane[:,2])]
allERDmean = [val for val in zip(tableauANOVA[:,0],tableauANOVA[:,1],tableauANOVA[:,2])]

allERDmediane = list(sum(allERDmediane,()))
allERDmean = list(sum(allERDmean,()))

df_mean = pd.DataFrame({'condition': np.tile(["pendule", "main", "mainvib"],23),
                   'sujet': np.repeat([0,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],3),#ATTENTION 4 EST EN FAIT 3 LOL
                      'ERD':allERDmean})
df_mediane = pd.DataFrame({'condition': np.tile(["pendule", "main", "mainvib"],23),
                   'sujet': np.repeat([0,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],3),#ATTENTION 4 EST EN FAIT 3 LOL
                      'ERD':allERDmediane})

df_mean.to_csv("../csv_files/H3_anova/"+elec+"_"+str(fmin)+"_"+str(fmax)+"Hz_redone_anova_long.csv")


df_2cond_mediane = df_mediane[df_mediane["condition"]!="mainvib"]
df_2cond_mediane_main = df_mediane[df_mediane["condition"]!="pendule"]

anovaMediane = AnovaRM(data=df_mediane, depvar='ERD', subject='sujet', within=['condition']).fit()
anovaMean = AnovaRM(data=df_mean, depvar='ERD', subject='sujet', within=['condition']).fit()
print(anovaMediane)
print(anovaMean)

anovaMediane_mainPendule = AnovaRM(data=df_2cond_mediane, depvar='ERD', subject='sujet', within=['condition']).fit()
print(anovaMediane_mainPendule)

anovaMediane_mainMainIllusion= AnovaRM(data=df_2cond_mediane_main, depvar='ERD', subject='sujet', within=['condition']).fit()
print(anovaMediane_mainMainIllusion)