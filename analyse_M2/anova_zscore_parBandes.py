# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:48:40 2022

@author: claire.dussard
"""
from functions.load_savedData import *
from handleData_subject import createSujetsData
from functions.load_savedData import *
from functions.frequencyPower_displays import *

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()

#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

liste_tfrMain = load_tfr_data_windows(liste_rawPathMain,"",True)
liste_tfrMainIllusion = load_tfr_data_windows(liste_rawPathMainIllusion,"",True)
liste_tfrPendule = load_tfr_data_windows(liste_rawPathPendule,"",True)

liste_tfr_pendule = liste_tfrPendule.copy()
liste_tfr_main = liste_tfrMain.copy()
liste_tfr_mainIllusion = liste_tfrMainIllusion.copy()

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
    return tableauANOVA,tableauANOVAmediane

tableauANOVA_c3SMR,tableauANOVAmediane_c3SMR = anova_data_elec(liste_tfr_main,liste_tfr_mainIllusion,liste_tfr_pendule,"logratio",12,15,2,25,"C3")


#ANOVA
from statsmodels.stats.anova import AnovaRM
allERDmediane = [val for val in zip(tableauANOVAmediane_c3SMR[:,0],tableauANOVAmediane_c3SMR[:,1],tableauANOVAmediane_c3SMR[:,2])]
allERDmean = [val for val in zip(tableauANOVA_c3SMR[:,0],tableauANOVA_c3SMR[:,1],tableauANOVA_c3SMR[:,2])]

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
# tableauANOVA_NoBL_seuil_mean = tableauANOVA
# pd.DataFrame(tableauANOVA_NoBL_seuil_mean).to_csv("../csv_files/ANOVA_C3_noBL_seuil/tableauANOVA_mean_Seuil_C3_8-30Hz.csv")

# tableauANOVA_NoBL_seuil_med = tableauANOVAmediane
# pd.DataFrame(tableauANOVA_NoBL_seuil_med).to_csv("../csv_files/ANOVA_C3_noBL_seuil/tableauANOVA_med_Seuil_C3_8-30Hz.csv")
