#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 16:16:55 2022

@author: claire.dussard
"""

from functions.load_savedData import *
from handleData_subject import createSujetsData
essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates = createSujetsData()

#pour se placer dans les donnees lustre
os.chdir("../../../../../../")
lustre_data_dir = "iss02/cenir/analyse/meeg/BETAPARK/_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)


liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

 
#ON REFAIT TOUT PROPREMENT 
#event_id = {'Essai_mainTactile':3}
event_id_mainIllusion = {'Essai_mainIllusion':3}
event_id_pendule={'Essai_pendule':4}  
event_id_main={'Essai_main':3}  

sujets_epochs_jetes_main = [
    [],[],[],[3],[2,4,5,6,7],[7],[],[6],[9,10],[8],[6],
    [1,6,8],[1,10],[9,10],[6,7,8,9,10],[3,6],[3,6,7],[4,10],[],[1,6],[],[9],[]
    ]

sujets_epochs_jetes_pendule = [
    [],[],[],[5],[1,7,10],[],[],[3,5,8,10],[],[5,10],[],
    [5,6],[4],[6,9],[],[9],[3,8,9],[],[],[1,6],[6],[3,9],[6,8]
    ]

sujets_epochs_jetes_mainIllusion = [
    [6],[1,3,6],[1,2],[],[5,6,8,9,10],[],[],[1,6,7,8],[6,7,8,9,10],[4,10],[1],
    [],[1,8,10],[10],[6,9],[9],[4,8,9],[4,8],[],[1,6],[],[1],[]
    ]
#================================================
#check data
# for i in range(23):
#     print("sujet "+str(i))
#     print("main")
#     print(10-len(sujets_epochs_jetes_main[i])==len(EpochDataMain[i]))
#     print("pendule")
#     print(10-len(sujets_epochs_jetes_pendule[i])==len(EpochDataPendule[i]))
#     print("mainIllusion")
#     print(10-len(sujets_epochs_jetes_mainIllusion[i])==len(EpochDataMainIllusion[i]))
#     print("\n")
# #sujets PB = 6&10 (4&8)
# for i in [4,8]:
#     print("sujet "+str(i))
#     print("main")
#     print(10-len(sujets_epochs_jetes_main[i]))
#     print(len(EpochDataMain[i]))
#     print("mainIllusion")
#     print(10-len(sujets_epochs_jetes_mainIllusion[i]))
#     print(len(EpochDataMainIllusion[i]))
    

#============================================

EpochDataMain = load_data_postICA_postdropBad(liste_rawPathMain,"")
EpochDataPendule = load_data_postICA_postdropBad(liste_rawPathPendule,"")
EpochDataMainIllusion = load_data_postICA_postdropBad(liste_rawPathMainIllusion,"")

montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
for epochs in EpochDataMain:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)
for epochs in EpochDataPendule:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)
for epochs in EpochDataMainIllusion:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)

def return_range_dispo_cond(epochs_jetes_cond):
    range_dispo = [i for i in range(1,11)]
    if len(epochs_jetes_cond)>0: 
        for item in epochs_jetes_cond:
            range_dispo.remove(item)
    return range_dispo

def compute_epochs_power(epochs):
    liste_puissance = []
    freqs = np.arange(3, 60, 1)  # frequencies from 2-35Hz
    n_cycles = freqs
    for i in range(len(epochs)):
        epoch = epochs[i]
        epoch_down = epoch.resample(250,npad='auto')
        print("computing power...")
        power_main_i = mne.time_frequency.tfr_morlet(epoch_down,freqs=freqs,n_cycles=n_cycles,return_itc=False)
        liste_puissance.append(power_main_i)
    return liste_puissance

def return_beta_values(liste_power):
    liste_beta_values = []
#crop time & frequency
    print(len(liste_power))
    for i in range(len(liste_power)):
        essai_i_cropped_8_30 = liste_power[i].copy().crop(fmin=8,fmax=30,tmin=-3,tmax=-1)#avant -2 ; 0
        power_essai_i = essai_i_cropped_8_30.data #on peut obtenir toutes les bandes de frequence en moyennant la dim 1, pas besoin de recropper
        power_C3_essai_i_8_30 = power_essai_i[11]
    
        val_i = power_C3_essai_i_8_30.mean()
        liste_beta_values.append(val_i)
    return liste_beta_values
    
def get_sujet_baselines(i):
#epochs non dispos par sujet compte entre 1 et 10
    S0i_jete_main = sujets_epochs_jetes_main[i]
    S0i_jete_pendule = sujets_epochs_jetes_pendule[i]
    S0i_jete_mainIllusion = sujets_epochs_jetes_mainIllusion[i]
    S0i_garde_main = return_range_dispo_cond(S0i_jete_main)
    S0i_garde_pendule = return_range_dispo_cond(S0i_jete_pendule)
    S0i_garde_mainIllusion = return_range_dispo_cond(S0i_jete_mainIllusion)
    
    print("garde main")
    print(S0i_garde_main)
    print("garde pendule")
    print(S0i_garde_pendule)
    print("garde main Illusion")
    print(S0i_garde_mainIllusion)
    epochsMain = EpochDataMain[i]
    epochsPendule = EpochDataPendule[i]
    epochsMainIllusion = EpochDataMainIllusion[i]
    
    liste_power_main = compute_epochs_power(epochsMain)
    liste_power_pendule = compute_epochs_power(epochsPendule)
    liste_power_mainIllusion = compute_epochs_power(epochsMainIllusion)
    valBetaBaseline_main = return_beta_values(liste_power_main)
    valBetaBaseline_pendule = return_beta_values(liste_power_pendule)
    valBetaBaseline_mainIllusion = return_beta_values(liste_power_mainIllusion)
    
    #valeurs par essai run moyenne
    values_run1_mi,values_run2_mi,num_essaisDispos_mi,epochs_moyennes_mi,array_tous_essais_mainIllusion = return_beta_values_meanRun(valBetaBaseline_mainIllusion,S0i_garde_mainIllusion)
    values_run1_m,values_run2_m,num_essaisDispos_m,epochs_moyennes_m,array_tous_essais_main = return_beta_values_meanRun(valBetaBaseline_main,S0i_garde_main)
    values_run1_p,values_run2_p,num_essaisDispos_p,epochs_moyennes_p,array_tous_essais_pendule = return_beta_values_meanRun(valBetaBaseline_pendule,S0i_garde_pendule)
    
    return S0i_garde_pendule,S0i_garde_main,S0i_garde_mainIllusion,valBetaBaseline_pendule,valBetaBaseline_main,valBetaBaseline_mainIllusion,\
num_essaisDispos_p,num_essaisDispos_m,num_essaisDispos_mi,\
epochs_moyennes_p,epochs_moyennes_m,epochs_moyennes_mi,\
array_tous_essais_main, array_tous_essais_mainIllusion,array_tous_essais_pendule


    
def get_matrix_essais_BL(nbSujets):
    array_essai_main = np.empty((nbSujets,5), dtype=float)
    array_essai_mainIllusion = np.empty((nbSujets,5), dtype=float)
    array_essai_pendule = np.empty((nbSujets,5), dtype=float)
    array_tousSujets_essais_main = np.empty((nbSujets,10), dtype=float)
    array_tousSujets_essais_mainIllusion = np.empty((nbSujets,10), dtype=float)
    array_tousSujets_essais_pendule = np.empty((nbSujets,10), dtype=float)
    for i in range(nbSujets):
        print("sujet "+str(i))
        S0i_garde_pendule,S0i_garde_main,S0i_garde_mainIllusion,\
        valBetaBaseline_pendule,valBetaBaseline_main,valBetaBaseline_mainIllusion,\
        num_essaisDispos_p,num_essaisDispos_m,num_essaisDispos_mi,\
        epochs_moyennes_p,epochs_moyennes_m,epochs_moyennes_mi,\
        array_tous_essais_main, array_tous_essais_mainIllusion,array_tous_essais_pendule=\
        get_sujet_baselines(i)#RAJOUTER LES VALEURS PAR SUJET TOUS ESSAIS
        print("essais dispo MI")
        print(num_essaisDispos_mi)
        #remplir array_tousSujets_essais_pendule

        array_tousSujets_essais_main[i,:] = array_tous_essais_main
        array_tousSujets_essais_mainIllusion[i,:] = array_tous_essais_mainIllusion
        array_tousSujets_essais_pendule[i,:] = array_tous_essais_pendule
        # for essai in range(1,6):
        #     print("essai "+str(essai))
        #     # if essai in num_essaisDispos_m:
        #     #     print(epochs_moyennes_m)
        #     #     print("i :"+str(i))
        #     #     array_essai_main[i][essai-1] = epochs_moyennes_m[num_essaisDispos_m.index(essai)]
        #     # else:
        #     #     array_essai_main[i][essai-1] = -1
        #     if essai in num_essaisDispos_mi:
        #         print(epochs_moyennes_mi)
        #         print("i :"+str(i))
        #         print(epochs_moyennes_mi[num_essaisDispos_mi.index(essai)])#nan
        #         array_essai_mainIllusion[i][essai-1] = epochs_moyennes_mi[num_essaisDispos_mi.index(essai)]#pb : 5 a index 4
        #     else : 
        #         array_essai_mainIllusion[i][essai-1] = -1
        #     # if essai in num_essaisDispos_p:
        #     #     print(epochs_moyennes_p)
        #     #     print("i :"+str(i))
        #     #     array_essai_pendule[i][essai-1] = epochs_moyennes_p[num_essaisDispos_p.index(essai)]
        #     # else : 
        #     #     array_essai_pendule[i][essai-1] = -1
                
    return array_tousSujets_essais_pendule,array_tousSujets_essais_mainIllusion,array_tousSujets_essais_main
                #array_essai_main,array_essai_mainIllusion,array_essai_pendule,
array_essai_main,array_essai_mainIllusion,array_essai_pendule,\
array_tousSujets_essais_pendule,array_tousSujets_essais_mainIllusion,array_tousSujets_essais_main =\
    get_matrix_essais_BL(2)#23
    

pd.DataFrame(array_tousSujets_essais_pendule).to_csv("tousEssais_pendule.csv")
pd.DataFrame(array_tousSujets_essais_mainIllusion).to_csv("tousEssais_mainIllusion.csv")
pd.DataFrame(array_tousSujets_essais_main).to_csv("tousEssais_main.csv")
#essaisMean
pd.DataFrame(array_essai_pendule).to_csv("meanEssais_pendule.csv")
pd.DataFrame(array_essai_mainIllusion).to_csv("meanEssais_mainIllusion.csv")
pd.DataFrame(array_essai_main).to_csv("meanEssais_main.csv")





def return_beta_values_meanRun(liste_beta_values,epochsDispo):
    epochs_run1 = [val for val in epochsDispo if val<6]
    epochs_run2 = [val for val in epochsDispo if val>5]
    values_run1 = liste_beta_values[0:len(epochs_run1)]
    values_run2 = liste_beta_values[-len(epochs_run2):]
    epochs_moyennes = []
    num_essaisDispos = [i for i in range(1,6)]
    array_tous_essais = np.empty((1,10), dtype=float)
    for j in range(1,11):
        if j in epochsDispo:
            #print("yes")
            index_essai = epochsDispo.index(j)#trouver ou est la valeur
            #print(index_essai)
            array_tous_essais[:,j-1] = liste_beta_values[index_essai]
        else:
            #print("no")
            array_tous_essais[:,j-1] = -1
    for i in range(1,6):
        print("i = "+str(i))
        if i in epochs_run1:
            index_run1 = epochs_run1.index(i)#obligatoire car valeurs manquantes
            if i+5 in epochs_run2:#2 runs dispo
                index_run2 = epochs_run2.index(i+5)
                print("deux valeurs dispos")
                print(values_run1[index_run1])
                print(values_run2[index_run2])
                moy = (values_run2[index_run2] + values_run1[index_run1])/2
                epochs_moyennes.append(moy)
            else:#run 1 dispo mais pas run2
                epochs_moyennes.append(values_run1[index_run1])
                print("run 1 mais pas 2 dispos")
                print(values_run1[index_run1])
        else:#run 1 pas dispo
            if i+5 in epochs_run2:#run 2 dispo mais pas 1
                print("run 2 mais pas 1 dispos")
                index_run2 = epochs_run2.index(i+5)
                epochs_moyennes.append(values_run2[index_run2])
            else:#aucun run dispo
                print("aucun run dispo")
                epochs_moyennes.append(-1.0)
                num_essaisDispos.remove(i)        
    return values_run1,values_run2,num_essaisDispos,epochs_moyennes,array_tous_essais

epochsDispo = S0i_garde_main
liste_beta_values = valBetaBaseline_main
yo = return_beta_values_meanRun(valBetaBaseline_main,S0i_garde_main)
raw_signal.plot(block=True)

#epochs dispos par sujet (deduits)
# i = 1
# S0i_garde_pendule,S0i_garde_main,S0i_garde_mainIllusion,\
# valBetaBaseline_pendule,valBetaBaseline_main,valBetaBaseline_mainIllusion,\
# num_essaisDispos_p,num_essaisDispos_m,num_essaisDispos_mi,\
# epochs_moyennes_p,epochs_moyennes_m,epochs_moyennes_mi = \
# get_sujet_baselines(i)
     
#     #plot 10 essais
# ax,fig = plt.subplots()
# plt.plot(S0i_garde_pendule,valBetaBaseline_pendule,label="pendule")
# plt.plot(S0i_garde_main,valBetaBaseline_main,label="main")
# plt.plot(S0i_garde_mainIllusion,valBetaBaseline_mainIllusion,label="mainIllusion")
# ax.legend()
  
# ax,fig = plt.subplots()
# plt.plot(num_essaisDispos_mi,epochs_moyennes_mi,label="mainIllusion")
# plt.plot(num_essaisDispos_m,epochs_moyennes_m,label="main")
# plt.plot(num_essaisDispos_p,epochs_moyennes_p,label="pendule")
# ax.legend()
# #ax.set_ylim([-0.35, 0.23])
# raw_signal.plot(block=True)

