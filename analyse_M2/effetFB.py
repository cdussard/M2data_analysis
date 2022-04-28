#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:11:23 2021

@author: claire.dussard
"""
import os 
import seaborn as sns
import pathlib
from handleData_subject import createSujetsData
from functions.load_savedData import *
from functions.preprocessData_eogRefait import *
import numpy as np 

nom_essai = "4"
essaisFeedbackSeul = ["pas_enregistre","sujet jeté",
"4","4","sujet jeté","4","4","4","4","MISSING","4","4",
"4","4","4","4","4","4","4","4-b","4","4","4","4","4","4"]

# essaisFeedbackSeul = [nom_essai for i in range(25)]

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()
sujetsPb = [0,9]
for sujetpb in sujetsPb:
    allSujetsDispo.remove(sujetpb)
liste_rawPathEffetFBseul = createListeCheminsSignaux(essaisFeedbackSeul,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
# plot les NFB a cote des rest
liste_rawPathMain = createListeCheminsSignaux(essaisMainSeule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)
liste_rawPathPendule = createListeCheminsSignaux(essaisPendule,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

#ordre inverse au milieu de la manip pr eviter effets ordre)
event_id_mainIllusion = {'Main illusion seule': 26}
event_id_pendule={'Pendule seul':23}  
event_id_main={'Main seule': 24}  

nbSujets = 2
SujetsDejaTraites = 0
rawPathEffetFBseul_sujets = liste_rawPathEffetFBseul[SujetsDejaTraites:SujetsDejaTraites+nbSujets]

listeEpochs_main,listeICA,listeEpochs_pendule,listeEpochs_mainIllusion = all_conditions_analysis_FBseul(allSujetsDispo,rawPathEffetFBseul_sujets,
                            event_id_main,event_id_pendule,event_id_mainIllusion,
                            0.1,1,90,[50,100],'Fz')


saveEpochsAfterICA_FBseul(listeEpochs_main,rawPathEffetFBseul_sujets,"main",True)
saveEpochsAfterICA_FBseul(listeEpochs_pendule,rawPathEffetFBseul_sujets,"pendule",True)
saveEpochsAfterICA_FBseul(listeEpochs_mainIllusion,rawPathEffetFBseul_sujets,"mainIllusion",True)
save_ICA_files(listeICA,rawPathEffetFBseul_sujets,True)

#=============== EPOCHS POWER ==========================
#load saved Data
EpochDataMain = load_data_postICA_preDropbad_effetFBseul(liste_rawPathEffetFBseul,"main",True)

EpochDataPendule = load_data_postICA_preDropbad_effetFBseul(liste_rawPathEffetFBseul,"pendule",True)

EpochDataMainIllusion = load_data_postICA_preDropbad_effetFBseul(liste_rawPathEffetFBseul,"mainIllusion",True)
#====================drop epochs with artefacts ======================
#display epochs, chose which to drop
initial_ref = 'Fz'
liste_epochs_averageRef_main = []
liste_epochs_averageRef_mainIllusion = []
liste_epochs_averageRef_pendule = []
channelsSansFz = ['Fp1', 'Fp2', 'F7', 'F3','F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10','T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5','CP1','CP2','CP6','TP10','P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2','HEOG','VEOG']

epochs = EpochDataMain
listeEpochs = liste_epochs_averageRef_main
for num_sujet in range(len(epochs)):
    #============== PLOT & DROP EPOCHS========================================
    #epochs[num_sujet].reorder_channels(channelsSansFz) #nathalie : mieux de jeter epochs avant average ref (sinon ça va baver partout artefact)
    #epochs[num_sujet].plot(n_channels=35,n_epochs=1) #select which epochs, which channels to drop
    #====================MAIN===============================================================
    epochs[num_sujet].info["bads"]=["FT9","FT10","TP9","TP10"]
    signalInitialRef = mne.add_reference_channels(epochs[num_sujet],initial_ref)
    averageRefSignal = signalInitialRef.set_eeg_reference('average')
    listeEpochs.append(averageRefSignal)

for epoch in EpochDataMain:
    epoch.info["bads"]=["FT9","FT10","TP9","TP10"]
    
rawTest.plot(block=True)
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

#est ce qu'il faudrait faire une BL averagée sur les 3 epochs puis plot les epochs ?
liste_power_main = plotSave_power_topo_cond(EpochDataMain,liste_rawPathEffetFBseul,3,85,"effet_mainSeule",250.)#needs to have set up the electrode montage before
liste_power_pendule = plotSave_power_topo_cond(EpochDataPendule,liste_rawPathEffetFBseul,3,85,"effet_penduleSeul",250.)
liste_power_mainIllusion = plotSave_power_topo_cond(EpochDataMainIllusion,liste_rawPathEffetFBseul,3,85,"effet_mainIllusion_Seule",250.)



#computing power and saving
def computepower(EpochData):
    liste_power_sujets = []
    i = 0
    freqs = np.arange(3, 85, 1)
    n_cycles = freqs 
    for epochs_sujet in EpochData:
        print("========================\nsujet"+str(i))
        epochData_sujet_down = epochs_sujet.resample(250., npad='auto') 
        print("downsampling...")
        power_sujet = mne.time_frequency.tfr_morlet(epochData_sujet_down,freqs=freqs,n_cycles=n_cycles,return_itc=False)
        print("computing power...")
        liste_power_sujets.append(power_sujet)
        i += 1
    return liste_power_sujets

liste_power_sujets_p = computepower(EpochDataPendule)
liste_power_sujets_m = computepower(EpochDataMain)
liste_power_sujets_mi = computepower(EpochDataMainIllusion)

len(liste_power_sujets_p) == len(liste_rawPathEffetFBseul)
save_tfr_data(liste_power_sujets_p,liste_rawPathEffetFBseul,"mainIllusion",True)

len(liste_power_sujets_m) == len(liste_rawPathEffetFBseul)
save_tfr_data(liste_power_sujets_m,liste_rawPathEffetFBseul,"main",True)
len(liste_power_sujets_mi) == len(liste_rawPathEffetFBseul)
save_tfr_data(liste_power_sujets_mi,liste_rawPathEffetFBseul,"pendule",True)

def print_effetFB_NFB_conditions(numSujet,scaleTopo,scaleTFR,my_cmap):
    #liste_rawPathMain les mauvais sujets ont deja ete drops !
    #============read the data==========================
    liste_tfr_m = load_tfr_data_windows(liste_rawPathEffetFBseul[numSujet:numSujet+1],"main",True) 
    liste_tfr_p = load_tfr_data_windows(liste_rawPathEffetFBseul[numSujet:numSujet+1],"pendule",True)
    liste_tfr_mi = load_tfr_data_windows(liste_rawPathEffetFBseul[numSujet:numSujet+1],"mainIllusion",True)
    
    liste_tfr_m_nfb = load_tfr_data_windows(liste_rawPathMain[numSujet:numSujet+1],"",True)
    liste_tfr_mi_nfb = load_tfr_data_windows(liste_rawPathMainIllusion[numSujet:numSujet+1],"",True)
    liste_tfr_p_nfb = load_tfr_data_windows(liste_rawPathPendule[numSujet:numSujet+1],"",True)
    
    #============baseline==========================
    dureePreBaseline = 3
    dureePreBaseline = - dureePreBaseline
    dureeBaseline = 2.0
    valeurPostBaseline = dureePreBaseline + dureeBaseline
    
    baseline = (dureePreBaseline, valeurPostBaseline)
    for tfr_p,tfr_m,tfr_mi in zip(liste_tfr_p,liste_tfr_m,liste_tfr_mi):
        tfr_p.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
        tfr_m.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
        tfr_mi.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
        
    for tfr_p,tfr_m,tfr_mi in zip(liste_tfr_p_nfb,liste_tfr_m_nfb,liste_tfr_mi_nfb):
        tfr_p.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
        tfr_m.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
        tfr_mi.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
     #============plot topomap and TFR C3==========================
     #create one image only
    fig1, axs1 = plt.subplots(2,3)
    fig2, axs2 = plt.subplots(2,3)
    for tfr_p,tfr_m,tfr_mi in zip(liste_tfr_p_nfb,liste_tfr_m_nfb,liste_tfr_mi_nfb):
        tfr_p.plot_topomap(fmin=8,fmax=30,tmin=2.5,tmax=26.5,vmin=-scaleTopo,vmax=scaleTopo,cmap=my_cmap,axes=axs1[0,0])
        tfr_m.plot_topomap(fmin=8,fmax=30,tmin=2.5,tmax=26.5,vmin=-scaleTopo,vmax=scaleTopo,cmap=my_cmap,axes=axs1[0,1])
        tfr_mi.plot_topomap(fmin=8,fmax=30,tmin=2.5,tmax=26.5,vmin=-scaleTopo,vmax=scaleTopo,cmap=my_cmap,axes=axs1[0,2])
        #TFR C3
        tfr_p.plot(picks="C3",fmax=40,vmin=-scaleTFR,vmax=scaleTFR,axes=axs2[0,0])#,tmin=2,tmax=25)
        tfr_m.plot(picks="C3",fmax=40,vmin=-scaleTFR,vmax=scaleTFR,axes=axs2[0,1])
        tfr_mi.plot(picks="C3",fmax=40,vmin=-scaleTFR,vmax=scaleTFR,axes=axs2[0,2])
        
    for tfr_p,tfr_m,tfr_mi in zip(liste_tfr_p,liste_tfr_m,liste_tfr_mi):
        tfr_p.plot_topomap(fmin=8,fmax=30,tmin=0,tmax=25,vmin=-scaleTopo,vmax=scaleTopo,cmap=my_cmap,axes=axs1[1,0])
        tfr_m.plot_topomap(fmin=8,fmax=30,tmin=0,tmax=25,vmin=-scaleTopo,vmax=scaleTopo,cmap=my_cmap,axes=axs1[1,1])
        tfr_mi.plot_topomap(fmin=8,fmax=30,tmin=0,tmax=25,vmin=-scaleTopo,vmax=scaleTopo,cmap=my_cmap,axes=axs1[1,2])
        #TFR C3
        tfr_p.plot(picks="C3",fmax=40,vmin=-scaleTFR,vmax=scaleTFR,axes=axs2[1,0])#,tmin=2,tmax=25)
        tfr_m.plot(picks="C3",fmax=40,vmin=-scaleTFR,vmax=scaleTFR,axes=axs2[1,1])
        tfr_mi.plot(picks="C3",fmax=40,vmin=-scaleTFR,vmax=scaleTFR,axes=axs2[1,2])
        
    rawTest.plot(block=True)
    return fig1,fig2

ScalesSujetsGraphes_8a30Hz = [0.3,0.38,0.28,0.4,#S00-06
                              0.28,0.28,0.4,0.34,#S07-11
                              0.18,0.24,0.35,0.25,0.32,#S12-16
                              0.22,0.4,0.24,0.16,0.3,0.4,0.26,0.4]#S17-21
           

def plotAllSujetsFB_NFB(sujetStart,sujetEnd): 
    for i in range(sujetStart,sujetEnd):
        listFiguresTopo = []
        listFiguresTFR = []
        numSujet = i
        multiScaleTopo = 1.7
        multiScaleTFR = 2.3
        scaleTFR = ScalesSujetsGraphes_8a30Hz[numSujet]*multiScaleTFR
        print(scaleTFR)
        scaleTopo= ScalesSujetsGraphes_8a30Hz[numSujet]*multiScaleTopo
        print(scaleTopo)
        my_cmap = discrete_cmap(13, 'RdBu_r')
        figTopo,figTFR = print_effetFB_NFB_conditions(numSujet,scaleTopo,scaleTFR,my_cmap)
        listFiguresTFR.append(figTFR)
        listFiguresTopo.append(figTopo)
    return listFiguresTFR,listFiguresTopo

listTFR,listTopo = plotAllSujetsFB_NFB(1,6)
listTFR2,listTopo2 = plotAllSujetsFB_NFB(7,13)
listTFR2,listTopo2 = plotAllSujetsFB_NFB(7,13)
    
#============= compute average power ==============
#check when we can do a baseline
rawTest = mne.io.read_raw_brainvision(liste_rawPathEffetFBseul[8],preload=True,eog=('HEOG', 'VEOG'))#,misc=('EMG'))#AJOUT DU MISC EMG
events = mne.events_from_annotations(rawTest)[0]
rawTest.plot(block=True)

#affichage condition repos de -4 a -1 
dureePreBaseline = 3
dureePreBaseline = - dureePreBaseline
dureeBaseline = 2.0
valeurPostBaseline = dureePreBaseline + dureeBaseline

baseline = (dureePreBaseline, valeurPostBaseline)
for tfr_p,tfr_m,tfr_mi in zip(liste_power_sujets_p,liste_power_sujets_m,liste_power_sujetsmi):
    tfr_p.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    tfr_m.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    tfr_mi.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    
    
    
nSujetsToPlot = 5
sujetsAlreadyPlotted = 16
subsetToPlot_p = liste_power_sujets_p[sujetsAlreadyPlotted:sujetsAlreadyPlotted+nSujetsToPlot]
subsetToPlot_m = liste_power_sujets_m[sujetsAlreadyPlotted:sujetsAlreadyPlotted+nSujetsToPlot]
subsetToPlot_mi = liste_power_sujets_mi[sujetsAlreadyPlotted:sujetsAlreadyPlotted+nSujetsToPlot]
for tfr_p,tfr_m,tfr_mi in zip(subsetToPlot_p,subsetToPlot_m,subsetToPlot_mi):
    tfr_p.plot_topomap(fmin=8,fmax=30,tmin=2,tmax=25)
    tfr_m.plot_topomap(fmin=8,fmax=30,tmin=2,tmax=25)
    tfr_mi.plot_topomap(fmin=8,fmax=30,tmin=2,tmax=25)
rawTest.plot(block=True)
scalesSujet =[0.35]


#================compute grand average===============================================
mode = 'logratio'
av_power_mainIllusion = mne.grand_average(liste_power_sujets_mi,interpolate_bads=False)
save_topo_data(av_power_mainIllusion,dureePreBaseline,valeurPostBaseline,"all_sujets",mode,"effet_mainIllusionSeule",False,1.0,24.0)#can be improved with boolean Params for alpha etcliste_tfr,interpolate_bads=True)
  
av_power_main = mne.grand_average(liste_power_sujets_m,interpolate_bads=False)
save_topo_data(av_power_main,dureePreBaseline,valeurPostBaseline,"all_sujets",mode,"effet_mainSeule",False,1.0,24.0)#can be improved with boolean Params for alpha etcliste_tfr,interpolate_bads=True)

av_power_pendule = mne.grand_average(liste_power_sujets_p,interpolate_bads=False)
save_topo_data(av_power_pendule,dureePreBaseline,valeurPostBaseline,"all_sujets",mode,"effet_penduleSeul",False,1.0,24.0)#can be improved with boolean Params for alpha etcliste_tfr,interpolate_bads=True)

v = 0.6
av_power_pendule.plot(picks=["C3"],fmax=40,vmin=-v,vmax=v)
av_power_main.plot(picks=["C3"],fmax=40,vmin=-v,vmax=v)
av_power_mainIllusion.plot(picks=["C3"],fmax=40,vmin=-v,vmax=v)
rawTest.plot(block=True)

v = 0.38
av_power_pendule.plot_topomap(fmin=8,fmax=30,tmin=2,tmax = 25,vmin=-v,vmax=v)
av_power_main.plot_topomap(fmin=8,fmax=30,tmin=2,tmax = 25,vmin=-v,vmax=v)
av_power_mainIllusion.plot_topomap(fmin=8,fmax=30,tmin=2,tmax = 25,vmin=-v,vmax=v)
rawTest.plot(block=True)



avpower_main_moins_mainIllusion = av_power_main - av_power_mainIllusion

avpower_main_moins_pendule = av_power_main - av_power_pendule

v= 0.6
avpower_main_moins_pendule.plot(picks=["C3"],fmax=40,vmin=-v,vmax=v)
avpower_main_moins_mainIllusion.plot(picks=["C3"],fmax=40,vmin=-v,vmax=v)
rawTest.plot(block=True)

save_topo_data(avpower_main_moins_pendule,dureePreBaseline,valeurPostBaseline,"all_sujets",mode,"effet_main-pendule",False,1.0,24.0)
save_topo_data(avpower_main_moins_mainIllusion,dureePreBaseline,valeurPostBaseline,"all_sujets",mode,"effet_main-mainIllusion",False,1.0,24.0)
