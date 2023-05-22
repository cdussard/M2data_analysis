# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 19:40:33 2022

@author: claire.dussard
"""

import os 
import seaborn as sns
import pathlib
from handleData_subject import createSujetsData
from functions.load_savedData import *
from functions.preprocessData_eogRefait import *
import numpy as np 
from plotPsd import * 

#create liste of file paths
essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData() 

nom_essai = "1-2"#on prend seulement le premier de 2 min pour l'instant faire simple

allSujetsDispo_rest = allSujetsDispo#S014 pas de marqueurs sur EEG, a voir + tard

allSujetsDispo_rest.append(4)

allSujetsDispo.pop()
        
#on commence au 3 (reste pbs noms)
liste_rawPath_rawRest = []
for num_sujet in allSujetsDispo_rest:
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
    liste_rawPath_rawRest.append(raw_path_sample)

print(liste_rawPath_rawRest)

#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

event_id={'Begin resting time':12}#TO DO  : modifier la duree des epochs pour avoir 2 min ? 
test = read_raw_data(liste_rawPath_rawRest[0:2])
for ts in test:
    events = mne.events_from_annotations(ts)
    print(events)
    
liste_epochsPreICA,liste_epochsSignal = pre_process_donnees(liste_rawPath_rawRest[21:23],1,0.1,90,[50,100],31,'Fz',event_id,110)#que 2 premiers sujets

listeICApreproc=[]
listeICA= []
for i in range(len(liste_epochsPreICA)):
    averageRefSignal_i,ICA_i = treat_indiv_data(liste_epochsPreICA[i],liste_epochsSignal[i],'Fz')
    listeICApreproc.append(averageRefSignal_i)
    listeICA.append(ICA_i)
    
    
ICA_rest = load_ICA_files_windows(liste_rawPath_rawRest,True)
for (epoch,ica) in zip(liste_epochsSignal,ICA_rest):
    ica.apply(epoch)
    
save_ICA_files(listeICA,liste_rawPath_rawRest[21:23],True)#a faire : certains sont vides ??
saveEpochsAfterICA_avantdropBad_windows(listeICApreproc,liste_rawPath_rawRest[21:23],True)

save_ICA_files(listeICA[0:2]+listeICA[3:],liste_rawPath_rawRest,True)#a faire : certains sont vides ??
saveEpochsAfterICA_avantdropBad_windows(listeICApreproc[0:2]+listeICApreproc[3:],liste_rawPath_rawRest,True)

epochDataRest = load_data_postICA_preDropbad_effetFBseul(liste_rawPath_rawRest[20:],"",True,True)
epochDataRest = listeICApreproc

initial_ref = "Fz"
liste_epochs_averageRef_rest = []
for epochSujet in epochDataRest:
    print(epochSujet)
    if epochSujet is not None:#range(len(epochDataMain_dropBad)):
    #============== PLOT & DROP EPOCHS========================================
        epochSujet.reorder_channels(channelsSansFz) #nathalie : mieux de jeter epochs avant average ref (sinon ça va baver partout artefact)
        epochSujet.plot(n_channels=35,n_epochs=1) #select which epochs, which channels to drop
        raw_signal.plot(block=True)
        #====================MI===============================================================
        epochSujet.info["bads"]=["FT9","FT10","TP9","TP10"]
        signalInitialRef_main = mne.add_reference_channels(epochSujet,initial_ref)
        averageRefSignal_main = signalInitialRef_main.set_eeg_reference('average')
        liste_epochs_averageRef_rest.append(averageRefSignal_main)
    else:
        print("None")
        liste_epochs_averageRef_rest.append(None)

saveEpochsAfterICA_apresdropBad_windows(liste_epochs_averageRef_rest,liste_rawPath_rawRest[21:23],True)

montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
for epochs in liste_epochs_averageRef_rest:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)
        
        
EpochData = liste_epochs_averageRef_rest
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
    
save_tfr_data(liste_power_sujets,liste_rawPath_rawRest[21:23],"",True)

liste_power_sujets = load_tfr_data_windows(liste_rawPath_rawRest,"",True)

#baseline par sujet
baseline = (-3,0)
for tfr in liste_power_sujets:
    tfr.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    
av_power_Rest = mne.grand_average(liste_power_sujets[0:4]+liste_power_sujets[5:],interpolate_bads=True)
av_power_Rest.save("../AV_TFR/all_sujets/rest-tfr.h5",overwrite=True)
av_power_Rest.save("../AV_TFR/all_sujets/rest_noBaseline-tfr.h5",overwrite=True)

my_cmap = discrete_cmap(13, 'RdBu_r')
av_power_Rest.plot_topomap(fmin=8,fmax=30,tmin=5,tmax=30,cmap = my_cmap)
av_power_Rest.plot_topomap(fmin=8,fmax=30,tmin=1,tmax=30,cmap = my_cmap,vmin=-0.3,vmax=0.3)
av_power_Rest.plot(picks="C3",fmin=3,fmax=40,vmin=-0.4,vmax=0.4)
raw_signal.plot(block=True)

#av_power_Rest.save("../AV_TFR/all_sujets/rest30s_1-2-tfr.h5",overwrite=True)

data = av_power_Rest.data
data_meanTps = np.mean(data,axis=2)
data_C3 = data_meanTps[11][:]
data_C4 = data_meanTps[13][:]


#from plotPsd import *

fig, ax = plt.subplots()
scaleMin = 0
scaleMax = 1e-9
plot_elec_cond(av_power_Rest,"C3","Rest",11,freqs,fig,ax,scaleMin,scaleMax,10,25)
plot_elec_cond(av_power_Rest,"C4","Rest",13,freqs,fig,ax,scaleMin,scaleMax,10,25)

raw_signal.plot(block=True)






