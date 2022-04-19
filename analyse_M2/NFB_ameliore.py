#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 14:45:03 2021

@author: claire.dussard
"""
import os 
import seaborn as sns
import pathlib
import mne
#necessite d'avoir execute handleData_subject.py, et load_savedData avant 
import numpy as np 
# importer les fonctions definies par moi 
from handleData_subject import createSujetsData
from functions.load_savedData import *

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()

#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
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

#=================================================================================================================================
                                                        #METHODE AVERAGE EPOCHS SUJETS
#=================================================================================================================================
from functions.preprocessData_eogRefait import *

nbSujets = 24
SujetsDejaTraites = 0
rawPath_main_sujets = liste_rawPathMain[SujetsDejaTraites:SujetsDejaTraites+nbSujets]
rawPath_pendule_sujets = liste_rawPathPendule[SujetsDejaTraites:SujetsDejaTraites+nbSujets]
rawPath_mainIllusion_sujets = liste_rawPathMainIllusion[SujetsDejaTraites:SujetsDejaTraites+nbSujets]

# rawPath_main_sujets.pop(5)
# rawPath_main_sujets.pop(4)

listeEpochs_main,listeICA_main,listeEpochs_pendule,listeICA_pendule,listeEpochs_mainIllusion,listeICA_mainIllusion = all_conditions_analysis(allSujetsDispo,rawPath_main_sujets,rawPath_pendule_sujets,rawPath_mainIllusion_sujets,
                            event_id_main,event_id_pendule,event_id_mainIllusion,
                            0.1,1,90,[50,100],'Fz')
        
#reload epochs to start from the beginning, without redoing the ICAs :
listeEpochs_main,listeEpochs_pendule,listeEpochs_mainIllusion = all_conditions_analysis_ICAload(allSujetsDispo,rawPath_main_sujets,rawPath_pendule_sujets,rawPath_mainIllusion_sujets,
                            event_id_main,event_id_pendule,event_id_mainIllusion,
                            0.1,1,90,[50,100],'Fz')    
    
#save tous les epochs 
saveEpochsAfterICA_avantdropBad(listeEpochs_main,rawPath_main_sujets)
saveEpochsAfterICA_avantdropBad(listeEpochs_pendule,rawPath_pendule_sujets)
saveEpochsAfterICA_avantdropBad(listeEpochs_mainIllusion,rawPath_mainIllusion_sujets)

#ENSUITE DROP DES EPOCHS
epochDataMain_dropBad = load_data_postICA_preDropbad(rawPath_main_sujets,"")
EpochDataPendule_dropBad = load_data_postICA_preDropbad(rawPath_pendule_sujets,"")
EpochDataMainIllusion_dropBad = load_data_postICA_preDropbad(rawPath_mainIllusion_sujets,"")

#display epochs, chose which to drop
initial_ref = 'Fz'
liste_epochs_averageRef_main = []
liste_epochs_averageRef_mainIllusion = []
liste_epochs_averageRef_pendule = []
channelsSansFz = ['Fp1', 'Fp2', 'F7', 'F3','F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10','T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5','CP1','CP2','CP6','TP10','P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2','HEOG','VEOG']

for num_sujet in [4]:#range(len(epochDataMain_dropBad)):
    #============== PLOT & DROP EPOCHS========================================
    epochDataMain_dropBad[num_sujet].reorder_channels(channelsSansFz) #nathalie : mieux de jeter epochs avant average ref (sinon ça va baver partout artefact)
    epochDataMain_dropBad[num_sujet].plot(n_channels=35,n_epochs=1) #select which epochs, which channels to drop
    # EpochDataPendule_dropBad[num_sujet].reorder_channels(channelsSansFz)
    # EpochDataPendule_dropBad[num_sujet].plot(n_channels=35,n_epochs=1)
    # EpochDataMainIllusion_dropBad[num_sujet].reorder_channels(channelsSansFz)
    # EpochDataMainIllusion_dropBad[num_sujet].plot(block=True,n_channels=35,n_epochs=1)
    #====================MAIN===============================================================
    epochDataMain_dropBad[num_sujet].info["bads"]=["FT9","FT10","TP9","TP10"]
    signalInitialRef_main = mne.add_reference_channels(epochDataMain_dropBad[num_sujet],initial_ref)
    averageRefSignal_main = signalInitialRef_main.set_eeg_reference('average')
    liste_epochs_averageRef_main.append(averageRefSignal_main)
    # #====================PENDULE===============================================================
    # EpochDataPendule_dropBad[num_sujet].info["bads"]=["FT9","FT10","TP9","TP10"]
    # signalInitialRef_pendule = mne.add_reference_channels(EpochDataPendule_dropBad[num_sujet],initial_ref)
    # averageRefSignal_pendule = signalInitialRef_pendule.set_eeg_reference('average')
    # liste_epochs_averageRef_pendule.append(averageRefSignal_pendule)
    # #====================MAIN ILLUSION==============================================================
    # EpochDataMainIllusion_dropBad[num_sujet].info["bads"]=["FT9","FT10","TP9","TP10"]
    # signalInitialRef_mainIllusion = mne.add_reference_channels(EpochDataMainIllusion_dropBad[num_sujet],initial_ref)
    # averageRefSignal_mainIllusion = signalInitialRef_mainIllusion.set_eeg_reference('average')
    # liste_epochs_averageRef_mainIllusion.append(averageRefSignal_mainIllusion)

raw_signal.plot(block=True)
    

i=num_sujet
for signal in liste_epochs_averageRef_main:
    path_sujet = rawPath_main_sujets[i]#attention ne marche que si on a les epochs dans l'ordre
    path_raccourci = str(path_sujet)[0:len(str(path_sujet))-4]
    path_raccourci_split = path_raccourci.split('/')
    directory = "../EPOCH_ICA_APRES_REF/" + path_raccourci_split[0] + "/"
    print(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    signal.save(directory+ path_raccourci_split[3] +"fif",overwrite=True)
    i += 1
print("done saving")
    
#save Epochs & ICA
saveEpochsAfterICA_apresdropBad(liste_epochs_averageRef_main,rawPath_main_sujets)
saveEpochsAfterICA_apresdropBad(liste_epochs_averageRef_pendule,rawPath_pendule_sujets)
saveEpochsAfterICA_apresdropBad(liste_epochs_averageRef_mainIllusion,rawPath_mainIllusion_sujets)

#===============================================

saveEpochsAfterICA(listeEpochs_main,rawPath_main_sujets)
save_ICA_files(listeICA_main,rawPath_main_sujets)
saveEpochsAfterICA(listeEpochs_pendule,rawPath_pendule_sujets)
save_ICA_files(listeICA_pendule,rawPath_pendule_sujets)
saveEpochsAfterICA(listeEpochs_mainIllusion,rawPath_mainIllusion_sujets)
save_ICA_files(listeICA_mainIllusion,rawPath_mainIllusion_sujets)

#load previous data
EpochDataMain = load_data_postICA_postdropBad(rawPath_main_sujets,"")

EpochDataPendule = load_data_postICA_postdropBad(rawPath_pendule_sujets,"")

EpochDataMainIllusion = load_data_postICA_postdropBad(rawPath_mainIllusion_sujets,"")

#===================set montage===IMPORTANT!!!!=======================
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
        
        
liste_power_main = plotSave_power_topo_cond(EpochDataMain,rawPath_main_sujets,3,85,"main",250.,1.5,25.5,allSujetsDispo)#needs to have set up the electrode montage before
liste_power_pendule = plotSave_power_topo_cond(EpochDataPendule,rawPath_pendule_sujets,3,85,"pendule",250.,1.5,25.5,allSujetsDispo)
liste_power_mainIllusion = plotSave_power_topo_cond(EpochDataMainIllusion,rawPath_mainIllusion_sujets,3,85,"mainIllusion",250.,1.5,25.5,allSujetsDispo)

# yo = plotSave_power_topo_cond_chooseScale(EpochDataMain,rawPath_main_sujets,EpochDataMainIllusion,rawPath_mainIllusion_sujets,EpochDataPendule,rawPath_pendule_sujets,3,85,250,1.5,25.5) 
 
ScalesSujetsGraphes_8a30Hz = [0.2,0.3,0.38,0.28,0.4,#S00-06
                              0.28,0.28,0.3,0.4,0.34,#S07-11
                              0.18,0.24,0.35,0.25,0.32,#S12-16
                              0.22,0.4,0.24,0.16,0.3,#S17-21
                              0.4,0.26,0.4,0.2,0.09]#S22-24 + echelle generale + echelle difference

ScalesSujetsGraphes_Theta = [0.24,0.28,0.3,0.24,0.24,#S00-06
                             0.28,0.24,0.28,0.33,0.2,#S07-11
                             0.2,0.3,0.28,0.24,0.28,#S12-16
                             0.22,0.24,0.18,0.22,0.28,#S17-21
                             0.24,0.22,0.18,0.12,0.09]#S22-24 + echelle generale

ScalesSujetsGraphes_Alpha = [0.44,0.32,0.38,0.25,0.34,#S00-06
                             0.26,0.32,0.38,0.4,0.34,#S07-11
                             0.18,0.3,0.35,0.26,0.38,#S12-16
                             0.24,0.28,0.3,0.28,0.4,#S17-21
                             0.4,0.4,0.35,0.28,0.09]#S22-24 + echelle generale

ScalesSujetsGraphes_LowBeta = [0.28,0.3,0.38,0.28,0.38,#S00-06
                               0.32,0.3,0.33,0.4,0.3,#S07-11
                               0.32,0.24,0.38,0.32,0.34,#S12-16
                               0.25,0.38,0.32,0.2,0.35,#S17-21
                               0.4,0.28,0.35,0.22,0.09]#S22-24 + echelle generale

ScalesSujetsGraphes_HighBeta = [0.28,0.3,0.3,0.3,0.4,#S00-06
                                0.3,0.3,0.3,0.33,0.38,#S07-11
                                0.28,0.35,0.38,0.35,0.25,#S12-16
                                0.28,0.4,0.24,0.27,0.32,#S17-21
                                0.36,0.28,0.35,0.2,0.09]#S22-24 + echelle generale

ScalesSujetsGraphes_LowGamma = [0.24,0.35,0.25,0.3,0.42,#S00-06
                                0.3,0.3,0.32,0.3,0.4,#S07-11
                                0.3,0.38,0.3,0.38,0.28,#S12-16
                                0.26,0.45,0.38,0.26,0.32,#S17-21
                                0.28,0.32,0.28,0.2,0.09]#S22-24 + echelle generale

ScalesSujetsGraphes_HighGamma = [0.24,0.35,0.25,0.3,0.42,#S00-06
                                 0.3,0.3,0.36,0.3,0.4,#S07-11
                                 0.38,0.4,0.3,0.38,0.28,#S12-16
                                 0.28,0.45,0.4,0.26,0.34,#S17-21
                                 0.34,0.38,0.25,0.2,0.09]#S22-24 + echelle generale
#======================Computing all powers and saving=================
liste_power_sujets = []
freqs = np.arange(3, 85, 1)
n_cycles = freqs 
i = 0
EpochData = EpochDataPendule

for epochs_sujet in EpochData:
    print("========================\nsujet"+str(i))
    epochData_sujet_down = epochs_sujet.resample(250., npad='auto') 
    print("downsampling...")
    power_sujet = mne.time_frequency.tfr_morlet(epochData_sujet_down,freqs=freqs,n_cycles=n_cycles,return_itc=False)
    print("computing power...")
    liste_power_sujets.append(power_sujet)
    i += 1

liste_power_sujets_main = liste_power_sujets
liste_power_sujets_pendule = liste_power_sujets
liste_power_sujets_mainIllusion = liste_power_sujets

save_tfr_data(liste_power_main,rawPath_main_sujets,"")

save_tfr_data(liste_power_pendule,rawPath_pendule_sujets,"")

save_tfr_data(liste_power_mainIllusion,rawPath_mainIllusion_sujets,"")
#load tfr data

#avoir pop(5) et pop(4) avant car sujets 6 & 7 epochs manquants ds conditions, en fait non pck le code marche 8-) "sujet x non traité"
liste_tfr = []
liste_tfr = liste_power_pendule#load_tfr_data(rawPath_main_sujets)
liste_tfr = load_tfr_data(rawPath_main_sujets,"")
#load_tfr_data(rawPath_main_sujets)
#load_tfr_data(rawPath_pendule_sujets)

Sujets2conditions = [2,7,9,10,11,15,18,19]
indSujets2cond = [1,5,7,8,12,15,16]
indSujets3cond = [0,2,3,4,6,9,10,11,12,13,14,17,18,19,20]

liste_tfr = [tfr[0] for tfr in liste_tfr] #pop 4 : sujet 6 et 12 : sujet 15
liste_tfr.pop(4)
liste_tfr.pop(12)
tfr_sujets2cond = [liste_tfr[i] for i in indSujets2cond]
tfr_sujets3cond = [liste_tfr[i] for i in indSujets3cond]
tfr_sujets3cond = tfr_sujets3cond[0:2]
#===================apply a baseline by subject before grand averaging=========================

dureePreBaseline = 3 #3
dureePreBaseline = - dureePreBaseline
dureeBaseline = 2.0 #2.0
valeurPostBaseline = dureePreBaseline + dureeBaseline

baseline = (dureePreBaseline, valeurPostBaseline)
for tfr in liste_tfr:
    tfr.apply_baseline(baseline=baseline, mode='logratio', verbose=None)

#================compute grand average===============================================
#get bad sujets out : pop 4 et 12

av_power_main = mne.grand_average(liste_tfr,interpolate_bads=True)


raw_signal.plot(block=True)#debloquer graphes

av_power_pendule = mne.grand_average(liste_tfr,interpolate_bads=True)
save_topo_data(av_power_pendule,dureePreBaseline,valeurPostBaseline,"all_sujets",mode,"pendule",False,1.5,25.5,23)#can be improved with boolean Params for alpha etcliste_tfr,interpolate_bads=True)

av_power_pendule_2cond = mne.grand_average(tfr_sujets2cond,interpolate_bads=True)
save_topo_data(avpower_main_moins_pendule,dureePreBaseline,valeurPostBaseline,"all_sujets",mode,"main-pendule",False,1.5,25.5,24)
save_topo_data(avpower_main_moins_mainIllusion,dureePreBaseline,valeurPostBaseline,"all_sujets",mode,"main-mainIllusion",False,1.5,25.5,24)#e_bads=True)# pop(8) = fichier avec plus petite taille permet de virer l'erreur
av_power_pendule_3cond = mne.grand_average(tfr_sujets3cond)#,interpolate_bads=True)

avpower_pendule3cond_Moins_2cond = av_power_pendule_3cond - av_power_pendule_2cond

##ValueError: operands could not be broadcast together with shapes (7825,) (9000,) 

av_power_mainIllusion = mne.grand_average(liste_tfr,interpolate_bads=True)

bads_mainI = ['FT9', 'TP9', 'TP10']
# for tfr in tfr_sujets3cond:
#     tfr.info["bads"] = bads_mainI pop len(3cond) - 1 ; 7 ; 6
    
av_power_mainIllusion2cond = mne.grand_average(tfr_sujets2cond,interpolate_bads=False)
av_power_mainIllusion3cond = mne.grand_average(tfr_sujets3cond,interpolate_bads=False)
avpower_mainI_3cond_Moins_2cond = av_power_mainIllusion3cond - av_power_mainIllusion2cond

mode = "logratio"
#=================== save the data images ==================================================
mode = "logratio"
save_topo_data(av_power_main,dureePreBaseline,valeurPostBaseline,"all_sujets",mode,"main",False,1.5,25.5,23)#can be improved with boolean Params for alpha etc
save_topo_data(av_power_main2cond,dureePreBaseline,valeurPostBaseline,"all_sujets2cond",mode,"main",False)
save_topo_data(av_power_main3cond,dureePreBaseline,valeurPostBaseline,"all_sujets3cond",mode,"main",False)

save_topo_data(avpower_main3cond_Moins_2cond,dureePreBaseline,valeurPostBaseline,"3-2all_sujets3-2cond",mode,"main",False)

#try with baseline computed before vs after
save_topo_data(av_power_main_noBL,dureePreBaseline,valeurPostBaseline,"all_sujets",mode,"main",True)

save_topo_data(av_power_pendule,dureePreBaseline,valeurPostBaseline,"all_sujets",mode,"pendule",False,1.5,25.5,23)#can be improved with boolean Params for alpha etc
save_topo_data(av_power_pendule_2cond,dureePreBaseline,valeurPostBaseline,"all_sujets2cond",mode,"pendule",False)#can be improved with boolean Params for alpha etc
save_topo_data(av_power_pendule_3cond,dureePreBaseline,valeurPostBaseline,"all_sujets3cond",mode,"pendule",False)

save_topo_data(avpower_pendule3cond_Moins_2cond,dureePreBaseline,valeurPostBaseline,"3-2all_sujets3-2cond",mode,"pendule",False)

save_topo_data(av_power_mainIllusion,dureePreBaseline,valeurPostBaseline,"all_sujets",mode,"mainIllusion",False,1.5,25.5,23)
save_topo_data(av_power_mainIllusion3cond,dureePreBaseline,valeurPostBaseline,"all_sujets3cond",mode,"mainIllusion",False)
save_topo_data(av_power_mainIllusion2cond,dureePreBaseline,valeurPostBaseline,"all_sujets2cond",mode,"mainIllusion",False)

save_topo_data(avpower_mainI_3cond_Moins_2cond,dureePreBaseline,valeurPostBaseline,"3-2all_sujets3-2cond",mode,"mainIllusion",False)

# liste_tfr.append(av_tfr1)
# liste_tfr.append(av_tfr2)
gdAverage = mne.grand_average(EpochDataMain,interpolate_bads=True)

liste_tfr = [tfr[0] for tfr in liste_tfr]
av_power_main = mne.grand_average(liste_tfr,interpolate_bads=True)

#=======================save the grand averaged power======================
av_power_main.save("../AV_TFR/all_sujets/main-tfr.h5",overwrite=True)
av_power_mainIllusion.save("../AV_TFR/all_sujets/mainIllusion-tfr.h5",overwrite=True)
avpower_main3cond_Moins_2cond.save("../AV_TFR/all_sujets/main3cond-2cond-tfr.h5",overwrite=True)
avpower_pendule3cond_Moins_2cond.save("../AV_TFR/all_sujets/pendule3cond-2cond-tfr.h5",overwrite=True)

av_power_pendule.save("../AV_TFR/all_sujets/pendule-tfr.h5",overwrite=True)
av_power_pendule_2cond.save("../AV_TFR/all_sujets/pendule_2cond-tfr.h5",overwrite=True)
#=============== load data average TFR============================
av_power_main =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main-tfr.h5")[0]
av_power_mainIllusion =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion-tfr.h5")[0]
av_power_pendule =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule-tfr.h5")[0]


def plot_C3_power(powerObject,timesTrial,timesVibration,fmin,fmax,vmin,vmax):
    fig,axes = plt.subplots()
    powerObject.plot(picks="C3",fmin=fmin,fmax=fmax,vmin=vmin,vmax=vmax,axes=axes)
    if timesTrial:
        axes.axvline(2, color='green', linestyle='--')
        axes.axvline(26.9, color='green', linestyle='--')
    if timesVibration:
        axes.axvline(6.5, color='black', linestyle='--')
        axes.axvline(8.3, color='black', linestyle='--')
        axes.axvline(12.7, color='black', linestyle='--')
        axes.axvline(14.5, color='black', linestyle='--')
        axes.axvline(18.92, color='black', linestyle='--')
        axes.axvline(20.72, color='black', linestyle='--')
        axes.axvline(25.22, color='black', linestyle='--')
        axes.axvline(27.02, color='black', linestyle='--')
    plt.show()
    return fig
vmax = 0.42
vmin = -vmax
figPendule = plot_C3_power(av_power_pendule,True,False,3,40,vmin,vmax)
figMain = plot_C3_power(av_power_main,True,False,3,40,vmin,vmax)
figMainIllusion = plot_C3_power(av_power_mainIllusion,True,True,3,40,vmin,vmax)

raw_signal.plot(block=True)
    

#================ Plot C3, Cz, C4 power#=====================================
#temps frequence
av_power = av_power_pendule
# av_power.plot([6], baseline=None, modliste_tfr_mainIllusion = liste_tfr.copy()e='logratio', title=av_power.ch_names[5],tmin=0.,tmax=28.5,vmin=-0.4,vmax=0.4)#c3
# av_power.plot([22], baseline=None, mode='logratio', title=av_power.ch_names[19],tmin=0.,tmax=28.5,vmin=-0.4,vmax=0.4)#cz
# av_power.plot([23], baseline=None, mode='logratio', title=av_power.ch_names[20],tmin=0.,tmax=28.5,vmin=-0.4,vmax=0.4)#c4
av_power.plot_topo(fmax=35)#cliquer sur C3,Cz,C4 et sauver graphes

raw_signal.plot(block=True)#debloquer graphes
# #=====================Compute difference between conditions======================================

avpower_main_moins_pendule = av_power_main - av_power_pendule

avpower_main_moins_mainIllusion = av_power_main - av_power_mainIllusion

avpower_main_moins_pendule.plot(picks="C3")
avpower_main_moins_mainIllusion.plot(picks="C3")
raw_signal.plot(block=True)

save_topo_data(avpower_main_moins_pendule,dureePreBaseline,valeurPostBaseline,"all_sujets",mode,"main-pendule",False,1.5,25.5,24)
save_topo_data(avpower_main_moins_mainIllusion,dureePreBaseline,valeurPostBaseline,"all_sujets",mode,"main-mainIllusion",False,1.5,25.5,24)

avpower_main_moins_pendule.plot_topo(fmax = 30)
raw_signal.plot(block=True)

avpower_main_moins_mainIllusion.plot_topo(fmax = 30)
raw_signal.plot(block=True)