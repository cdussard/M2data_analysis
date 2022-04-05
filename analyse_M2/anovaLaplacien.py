#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:55:39 2022

@author: claire.dussard
"""

liste_tfrMain = load_tfr_data(rawPath_main_sujets,"C3C4laplacien")
liste_tfr_main = liste_tfrMain.copy()
liste_tfrMainIllusion = load_tfr_data(rawPath_mainIllusion_sujets,"C3C4laplacien")
liste_tfr_mainIllusion = liste_tfrMainIllusion.copy()
liste_tfrPendule = load_tfr_data(rawPath_pendule_sujets,"C3C4laplacien")
liste_tfr_pendule = liste_tfrPendule.copy()
#compute baseline
baseline = (-2,0)
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

tableauANOVA_apresBaseline = tableauANOVA
pd.DataFrame(tableauANOVA_apresBaseline).to_csv("tableauANOVA_apresBaseline_C3laplacien_8-30Hz.csv")
tableauANOVA_avantBaseline = tableauANOVA
pd.DataFrame(tableauANOVA_avantBaseline).to_csv("tableauANOVA_avantBaseline_C3laplacien-30Hz.csv")

pd.DataFrame(tableauANOVAmediane).to_csv("tableauANOVAmediane_C3laplacien-30Hz.csv")


from statsmodels.stats.anova import AnovaRM
allERDmediane = [val for val in zip(tableauANOVAmediane[:,0],tableauANOVAmediane[:,1],tableauANOVAmediane[:,2])]
allERDmean = [val for val in zip(tableauANOVA_apresBaseline[:,0],tableauANOVA_apresBaseline[:,1],tableauANOVA_apresBaseline[:,2])]

allERDmediane = list(sum(allERDmediane,()))
allERDmean = list(sum(allERDmean,()))

df_mean = pd.DataFrame({'condition': np.tile([1, 2, 3],23),
                   'sujet': np.repeat([0,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],3),
                      'ERD':allERDmean})
df_mediane = pd.DataFrame({'condition': np.tile([1, 2, 3],23),
                   'sujet': np.repeat([0,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],3),
                      'ERD':allERDmediane})
df_2cond_mediane = df_mediane[df_mediane["condition"]!=3]
df_2cond_mediane_main = df_mediane[df_mediane["condition"]!=1]

anovaMediane = AnovaRM(data=df_mediane, depvar='ERD', subject='sujet', within=['condition']).fit()
anovaMean = AnovaRM(data=df_mean, depvar='ERD', subject='sujet', within=['condition']).fit()
print(anovaMediane)
print(anovaMean)

anovaMediane_mainPendule = AnovaRM(data=df_2cond_mediane, depvar='ERD', subject='sujet', within=['condition']).fit()
print(anovaMediane_mainPendule)

anovaMediane_mainMainIllusion= AnovaRM(data=df_2cond_mediane_main, depvar='ERD', subject='sujet', within=['condition']).fit()
print(anovaMediane_mainMainIllusion)