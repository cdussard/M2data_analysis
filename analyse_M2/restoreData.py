#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 15:58:46 2021

@author: claire.dussard
"""
#get the wrong data file and segment it using the event codes

sample_data_dir = pathlib.Path("sub-S006/ses-20210426/eeg")
raw_path_sample = sample_data_dir/("BETAPARK_2021-04-19_5-2-c.vhdr")

raw_signal = mne.io.read_raw_brainvision(raw_path_sample,preload=False,eog=('HEOG', 'VEOG'))

events = mne.events_from_annotations(raw_signal)[0]

#emplacements 255
eventStimCodes = events[:,2]

indices255 = np.where(eventStimCodes==255)[0]
#emplacements 29
indices29 = np.where(eventStimCodes==29)[0]
#pour segmenter events et les donner ensuite pour les ecrire

event_id = {'start condition':255}
#trouver la duree des fichiers
#fichier 1 :  entre 1er 255 et 1er 29 (exclut le temps enregistre entre les deux conditions)
events255 = events[events[:,2]==255,]
tps_premier255 = events255[0][0]
tps_deuxieme255 = events255[1][0]
dureeCondition1 = (tps_premier29 - tps_premier255)/1000 #temps en s
#fichier 2 : entre 2e 255 et 2e 29
events29 = events[events[:,2]==29,]
tps_premier29 = events29[0][0]
tps_deuxieme29 = events29[1][0]
dureeCondition2 = (tps_deuxieme29 - tps_deuxieme255 )/1000 #temps en s


rawCondition1 = raw_signal.copy().crop(tmin=0, tmax=dureeCondition1)
rawCondition2 = raw_signal.copy().crop(tmin=tps_deuxieme255/1000, tmax=tps_deuxieme29/1000)
#check events : 
print(mne.events_from_annotations(rawCondition1))
print(mne.events_from_annotations(rawCondition2))

from pybv import write_brainvision
sfreq = rawCondition1.info["sfreq"]
ch_names = rawCondition1.info["ch_names"]
condition1_data = rawCondition1.get_data()
condition2_data = rawCondition2.get_data()

#drop middle column of events 
eventsCondition1 = np.delete(mne.events_from_annotations(rawCondition1)[0], 1, 1)
eventsCondition2 = np.delete(mne.events_from_annotations(rawCondition2)[0], 1, 1)
#pb les dates des evenements doivent etre decalees, on les a enregistre avec celles du fichier d'avant 
condition2_colonnetimeFixed = mne.events_from_annotations(rawCondition2)[0][:,0] - tps_deuxieme255
eventsCondition2[:,0] = condition2_colonnetimeFixed
#check time first event is 0
print(eventsCondition2[0][0]==0)
#write the two files
write_brainvision(data=condition1_data, sfreq=sfreq, ch_names=ch_names,
                  fname_base= "BETAPARK_2021-04-19_5-2-c_exportMNE", folder_out="S06",
                  events=eventsCondition1,overwrite=True)

write_brainvision(data=condition2_data, sfreq=sfreq, ch_names=ch_names,
                  fname_base="BETAPARK_2021-04-19_6-2_exportMNE", folder_out="S06",
                  events=eventsCondition2,overwrite=True)

#===========================read the data and check if we have the artefacts

raw_path_sample2 = ("S06/BETAPARK_2021-04-19_6-2_exportMNE.vhdr")

raw_signal2 = mne.io.read_raw_brainvision(raw_path_sample2,preload=False,eog=('HEOG', 'VEOG'))

event_id_main={'Essai_main':3}  

raw_path_sample = sample_data_dir/("BETAPARK_2021-04-19_6-2.vhdr")
raw_signal_artefact = mne.io.read_raw_brainvision(raw_path_sample,preload=False,eog=('HEOG', 'VEOG'))

epochs_corrige = mne.Epochs(raw_signal2,mne.events_from_annotations(raw_signal2)[0],event_id_main,tmin=-5,tmax = 35,baseline=None, preload=False)
epochsini = mne.Epochs(raw_signal_artefact,mne.events_from_annotations(raw_signal_artefact)[0],event_id_main,tmin=-5,tmax = 35,baseline=None, preload=False)
epochs_corrige.plot()
epochsini.plot(block=True)