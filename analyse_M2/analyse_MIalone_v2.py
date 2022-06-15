# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:02:09 2022

@author: claire.dussard
"""

import os 
import seaborn as sns
import pathlib
from handleData_subject import createSujetsData
from functions.load_savedData import *
from functions.preprocessData_eogRefait import *
import numpy as np 

#create liste of file paths
essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData() 

nom_essai = "2-2"

allSujetsDispo_MI = allSujetsDispo[2:]

           
        
#on commence au 3 (reste pbs noms)
liste_rawPath_rawMIalone = []
for num_sujet in allSujetsDispo_MI:
    print("sujet n° "+str(num_sujet))
    nom_sujet = listeNumSujetsFinale[num_sujet]
    if num_sujet in SujetsPbNomFichiers:
        if num_sujet>0:
            date_sujet = '19-04-2021'
        else:
            date_sujet = '15-04-2021'
    else:
        date_sujet = dates[num_sujet]
    date_nom_fichier = date_sujet[-4:]+"-"+date_sujet[3:5]+"-"+date_sujet[0:2]+"_"
    dateSession = listeDatesFinale[num_sujet]
    sample_data_loc = listeNumSujetsFinale[num_sujet]+"/"+listeDatesFinale[num_sujet]+"/eeg"
    sample_data_dir = pathlib.Path(sample_data_loc)
    raw_path_sample = sample_data_dir/("BETAPARK_"+ date_nom_fichier + nom_essai+".vhdr")#il faudrait recup les epochs et les grouper ?
    liste_rawPath_rawMIalone.append(raw_path_sample)

print(liste_rawPath_rawMIalone)


#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

event_id={'Motor imagery alone':21}
liste_epochsPreICA,liste_epochsSignal = pre_process_donnees(liste_rawPath_rawMIalone,1,0.1,90,[50,100],31,'Fz',event_id,)#que 2 premiers sujets

listeICApreproc=[]
listeICA= []
for i in range(len(liste_epochsPreICA)):
    averageRefSignal_i,ICA_i = treat_indiv_data(liste_epochsPreICA[i],liste_epochsSignal[i],'Fz')
    listeICApreproc.append(averageRefSignal_i)
    listeICA.append(ICA_i)
    
#save tous les epochs 
save_ICA_files(listeICA,liste_rawPath_rawMIalone,True)
saveEpochsAfterICA_avantdropBad_windows(listeICApreproc,liste_rawPath_rawMIalone,True)

epochDataMIalone = load_data_postICA_preDropbad_effetFBseul(liste_rawPath_rawMIalone,"",True,True)

initial_ref = "Fz"
liste_epochs_averageRef_MI = []
for num_sujet in range(len(allSujetsDispo_MI)):#range(len(epochDataMain_dropBad)):
    #============== PLOT & DROP EPOCHS========================================
    epochDataMIalone[num_sujet].reorder_channels(channelsSansFz) #nathalie : mieux de jeter epochs avant average ref (sinon ça va baver partout artefact)
    epochDataMIalone[num_sujet].plot(n_channels=35,n_epochs=1) #select which epochs, which channels to drop
    raw_signal.plot(block=True)
    #====================MI===============================================================
    epochDataMIalone[num_sujet].info["bads"]=["FT9","FT10","TP9","TP10"]
    signalInitialRef_main = mne.add_reference_channels(epochDataMIalone[num_sujet],initial_ref)
    averageRefSignal_main = signalInitialRef_main.set_eeg_reference('average')
    liste_epochs_averageRef_MI.append(averageRefSignal_main)


raw_signal.plot(block=True)

saveEpochsAfterICA_apresdropBad_windows(liste_epochs_averageRef_MI,liste_rawPath_rawMIalone,True)

montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
for epochs in liste_epochs_averageRef_MI:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)
        
        
EpochData = liste_epochs_averageRef_MI
liste_power_sujets = []
freqs = np.arange(3, 85, 1)
n_cycles = freqs 
i = 0
for epochs_sujet in EpochData:
    print("========================\nsujet"+str(i))
    epochData_sujet_down = epochs_sujet.resample(250., npad='auto') 
    print("downsampling...")
    power_sujet = mne.time_frequency.tfr_morlet(epochData_sujet_down,freqs=freqs,n_cycles=n_cycles,return_itc=False)
    print("computing power...")
    liste_power_sujets.append(power_sujet)
    i += 1

save_tfr_data(liste_power_sujets,liste_rawPath_rawMIalone,"",True)

listeRaw = read_raw_data(liste_rawPath_rawMIalone[0:2])
events = mne.events_from_annotations(listeRaw[0])[0]
print(events)
events = mne.events_from_annotations(listeRaw[1])[0]
print(events)

baseline = (-1, -0.1)
for tfr in liste_power_sujets:
    tfr.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    
av_power_MIalone = mne.grand_average(liste_power_sujets,interpolate_bads=True)

my_cmap = discrete_cmap(13, 'RdBu_r')
av_power_MIalone.plot_topomap(fmin=8,fmax=30,tmin=1,tmax=26.5,cmap = my_cmap)
av_power_MIalone.plot_topomap(fmin=8,fmax=30,tmin=1,tmax=26.5,cmap = my_cmap,vmin=-0.3,vmax=0.3)
av_power_MIalone.plot(picks="C3",fmin=3,fmax=40,vmin=-0.4,vmax=0.4)
raw_signal.plot(block=True)

av_power_main.save("../AV_TFR/all_sujets/MIalone-tfr.h5",overwrite=True)

data = av_power_MIalone.data
data_meanTps = np.mean(data,axis=2)
data_C3 = data_meanTps[11][:]
data_C4 = data_meanTps[13][:]


#left motor cortex
fig, ax = plt.subplots()
scaleMin = -0.35
scaleMax = 0.05
plot_elec_cond(av_power_MIalone,"C3","MI_alone",11,freqs,fig,ax,scaleMin,scaleMax)
plot_elec_cond(av_power_MIalone,"C4","MI_alone",13,freqs,fig,ax,scaleMin,scaleMax)

#compare with NFB data
scaleMin = -0.35
plot_allElec(av_power_pendule,"pendule",["C3","C4"],[11,13],scaleMin,scaleMax,freqs)
plot_allElec(av_power_main,"main",["C3","C4"],[11,13],scaleMin,scaleMax,freqs)
plot_allElec(av_power_mainIllusion,"mainIllusion",["C3","C4"],[11,13],scaleMin,scaleMax,freqs)

raw_signal.plot(block=True)
