# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 16:26:54 2023

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
from functions.preprocessData_eogRefait import *

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()

#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

liste_rawPathMainIllusion = createListeCheminsSignaux(essaisMainIllusion,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

nbSujets = 24
SujetsDejaTraites = 0
sujetsSansVibrations = [1,5,7,9,13,16,17,20]
sujetsSansVibrations.reverse()
rawPath_mainIllusion_sujets = liste_rawPathMainIllusion[SujetsDejaTraites:SujetsDejaTraites+nbSujets]
for suj in sujetsSansVibrations:
    rawPath_mainIllusion_sujets.pop(suj)
    
#EpochDataMainIllusion = load_data_postICA_postdropBad_windows(rawPath_mainIllusion_sujets,"",True)

# suj_0 = EpochDataMainIllusion[0]

# mne.events_from_annotations(EpochDataMainIllusion[0])

listeRaw_mainIllusion = read_raw_data(liste_rawPathMainIllusion)
i = 0
for signal in listeRaw_mainIllusion:
    print(i)
    events_withVib = mne.events_from_annotations(signal) #print(events)
    print(len(np.where(events_withVib[0][:,2]==1)[0]))
    i += 1
# listeRaw_mainIllusion[2].plot()

# events_woVib = mne.events_from_annotations(listeRaw_mainIllusion[1]) #print(events)
# print(events_woVib)

event_id_vib={'Vibration':1}  
liste_epochsPreICA,liste_epochsSignal = pre_process_donnees(rawPath_mainIllusion_sujets,0.1,1,90,[50,100],None,'Fz',event_id_vib,4)

listeEpochs = []
listeICA= []

for i in range(len(rawPath_mainIllusion_sujets)):
    print("\n===========Sujet S "+str(allSujetsDispo[i])+"========================\n")
    #================Compute ICA on runs=================================================================
    epochs,ICA = treat_indiv_data(liste_epochsSignal[i],liste_epochsPreICA[i],'Fz') 
    listeEpochs.append(epochs)
    listeICA.append(ICA)
    
    
saveEpochsAfterICA_avantdropBad_windows_blink(listeEpochs,rawPath_mainIllusion_sujets,True)

channelsSansFz = ['Fp1', 'Fp2', 'F7', 'F3','F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10','T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5','CP1','CP2','CP6','TP10','P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2','HEOG','VEOG']
initial_ref = "Fz"
liste_epochs = []
for num_sujet in range(len(listeEpochs)):#range(len(epochDataMain_dropBad)):
    #============== PLOT & DROP EPOCHS========================================
    listeEpochs[num_sujet].reorder_channels(channelsSansFz) #nathalie : mieux de jeter epochs avant average ref (sinon ça va baver partout artefact)
    listeEpochs[num_sujet].plot(n_channels=35,n_epochs=1) #select which epochs, which channels to drop
    raw_signal.plot(block=True)
    #====================MI===============================================================
    listeEpochs[num_sujet].info["bads"]=["FT9","FT10","TP9","TP10"]
    liste_epochs.append(listeEpochs[num_sujet])
    
    
saveEpochsAfterICA_apresdropBad_windows_blink(liste_epochs,rawPath_mainIllusion_sujets,True)
save_ICA_files_blink(listeICA,rawPath_mainIllusion_sujets,True)


liste_signaux_loades = load_data_postICA_postdropBad_windows_blink(rawPath_mainIllusion_sujets,"blink",True)

#now what

montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
for epochs in liste_signaux_loades:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)
    
    
liste_power_sujets = []
freqs = np.arange(3, 40, 1)
n_cycles = freqs 
i = 0
EpochData = liste_signaux_loades

for epochs_sujet in EpochData:
    print("========================\nsujet"+str(i))
    epochData_sujet_down = epochs_sujet.resample(250., npad='auto') 
    print("downsampling...")
    power_sujet = mne.time_frequency.tfr_morlet(epochData_sujet_down,freqs=freqs,n_cycles=n_cycles,return_itc=False)
    print("computing power...")
    liste_power_sujets.append(power_sujet)
    i += 1

av_power_vib = mne.grand_average(liste_power_sujets,interpolate_bads=True)
av_power_vib.save("../AV_TFR/all_sujets/vibNFB_noBL-tfr.h5",overwrite=True)

my_cmap = discrete_cmap(13, 'Reds')
v = 6e-10
av_power_vib.plot_topomap(fmin=8,fmax=30,tmin=0,tmax=2,vmin=0,vmax=v,cmap=my_cmap,colorbar=True)
v = 1.4e-9
av_power_vib.plot_topomap(fmin=3,fmax=7,tmin=0,tmax=2,vmin=0,vmax=v,cmap=my_cmap,colorbar=True)

vmin = 0
vmax = 1e-9
av_power_vib.plot(picks=["C3"],tmin=-1,tmax=3,vmin=vmin,vmax=vmax,cmap=my_cmap,colorbar=True)
av_power_vib.plot(picks=["FC1"],tmin=-1,tmax=3,vmin=vmin,vmax=vmax,cmap=my_cmap,colorbar=True)

dureePreBaseline = 1 #3
dureePreBaseline = - dureePreBaseline
dureeBaseline = 1.0 #2.0
valeurPostBaseline = dureePreBaseline + dureeBaseline

baseline = (dureePreBaseline, valeurPostBaseline)
for tfr in liste_power_sujets:
    tfr.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
    
    
    
av_power_vib_bl = mne.grand_average(liste_power_sujets,interpolate_bads=True)
av_power_vib_bl.save("../AV_TFR/all_sujets/vibNFB_BL-tfr.h5",overwrite=True)
av_power_vib_bl = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/vibNFB_BL-tfr.h5")[0]

my_cmap = discrete_cmap(13, 'RdBu')
my_cmap_rev = my_cmap.reversed()
vmax = 0.08
vmin = -vmax
av_power_vib_bl.plot_topomap(fmin=8,fmax=30,tmin=0,tmax=2,vlim=(vmin,vmax),cmap=my_cmap_rev,colorbar=True)
av_power_vib_bl.plot_topomap(fmin=3,fmax=7,tmin=0,tmax=2,vlim=(vmin,vmax),cmap=my_cmap_rev,colorbar=True)


vmax = 0.2
vmin = -vmax
av_power_vib_bl.plot(picks=["C3"],tmin=-1,tmax=3,vmin=vmin,vmax=vmax,cmap=my_cmap_rev,colorbar=True)
av_power_vib_bl.plot(picks=["FC1"],tmin=-1,tmax=3,vmin=vmin,vmax=vmax,cmap=my_cmap_rev,colorbar=True)



#=============donnees de vibration=====================
nom_essai = "4"
essaisFeedbackSeul = ["pas_enregistre","sujet jeté",
"4","4","sujet jeté","4","4","4","4","MISSING","4","4",
"4","4","4","4","4","4","4","4-b","4","4","4","4","4","4"]

essaisFeedbackSeul = [nom_essai for i in range(25)]

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()
sujetsPb = [0,9,15]
for sujetpb in sujetsPb:
    allSujetsDispo.remove(sujetpb)
liste_rawPathEffetFBseul = createListeCheminsSignaux(essaisFeedbackSeul,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

    
#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

listeRaw_effetFB = read_raw_data(liste_rawPathEffetFBseul)
i = 0
listeBoolPremierVib = []
for signal in listeRaw_mainIllusion:
    print(i)
    events_withVib = mne.events_from_annotations(signal) #print(events)
    index_startMI = np.where(events_withVib[0][:,2]==26)[0][0]
    good_indexes_subset = events_withVib[0][index_startMI:index_startMI+21]
    print(len(np.where(good_indexes_subset[:,2]==1)[0]))
    premierVibIll = np.where(events_withVib[0][:,2]==26)[0][0] < np.where(events_withVib[0][:,2]==25)[0][0]
    listeBoolPremierVib.append(premierVibIll)
    i += 1
#il faut qu'on sache si on prend les 4 premiers ou pas pour l'epoching



event_id_vib={'Vibration':1}  
liste_epochsPreICA,liste_epochsSignal = pre_process_donnees(liste_rawPathEffetFBseul,0.1,1,90,[50,100],None,'Fz',event_id_vib,4)
i = 0
for epoch_ica,epoch_signal in zip(liste_epochsPreICA,liste_epochsSignal):
    if listeBoolPremierVib[i] == True:
        liste_epochsPreICA[i] = epoch_ica[0:4]#first 4 vibrations
        liste_epochsSignal[i] = epoch_signal[0:4]
        print("done")
    elif listeBoolPremierVib[i] == False:
        print("done false")
        liste_epochsPreICA[i] = epoch_ica[4:]#last 4 vibrations
        liste_epochsSignal[i] = epoch_signal[4:]
    i += 1 
        

listeEpochs = []
listeICA= []

for i in range(len(liste_rawPathEffetFBseul)):
    print("\n===========Sujet S "+str(allSujetsDispo[i])+"========================\n")
    #================Compute ICA on runs=================================================================
    epochs,ICA = treat_indiv_data(liste_epochsSignal[i],liste_epochsPreICA[i],'Fz') 
    listeEpochs.append(epochs)
    listeICA.append(ICA)
    
    
saveEpochsAfterICA_avantdropBad_windows_blink(listeEpochs,liste_rawPathEffetFBseul,True)

channelsSansFz = ['Fp1', 'Fp2', 'F7', 'F3','F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10','T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5','CP1','CP2','CP6','TP10','P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2','HEOG','VEOG']
initial_ref = "Fz"
liste_epochs = []
for num_sujet in range(len(listeEpochs)):#range(len(epochDataMain_dropBad)):
    #============== PLOT & DROP EPOCHS========================================
    listeEpochs[num_sujet].reorder_channels(channelsSansFz) #nathalie : mieux de jeter epochs avant average ref (sinon ça va baver partout artefact)
    listeEpochs[num_sujet].plot(n_channels=35,n_epochs=1) #select which epochs, which channels to drop
    raw_signal.plot(block=True)
    #====================MI===============================================================
    listeEpochs[num_sujet].info["bads"]=["FT9","FT10","TP9","TP10"]
    liste_epochs.append(listeEpochs[num_sujet])
    
    
saveEpochsAfterICA_apresdropBad_windows_blink(liste_epochs,liste_rawPathEffetFBseul,True)
save_ICA_files_blink(listeICA,liste_rawPathEffetFBseul,True)


liste_signaux_loades = load_data_postICA_postdropBad_windows_blink(liste_rawPathEffetFBseul,"blink",True)

#now what

montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
for epochs in liste_signaux_loades:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)
    
    
liste_power_sujets = []
freqs = np.arange(3, 40, 1)
n_cycles = freqs 
i = 0
EpochData = liste_signaux_loades

for epochs_sujet in EpochData:
    print("========================\nsujet"+str(i))
    epochData_sujet_down = epochs_sujet.resample(250., npad='auto') 
    print("downsampling...")
    power_sujet = mne.time_frequency.tfr_morlet(epochData_sujet_down,freqs=freqs,n_cycles=n_cycles,return_itc=False)
    print("computing power...")
    liste_power_sujets.append(power_sujet)
    i += 1

av_power_vib_effet = mne.grand_average(liste_power_sujets,interpolate_bads=True)
av_power_vib_effet.save("../AV_TFR/all_sujets/effetVibSeule_noBL-tfr.h5",overwrite=True)

my_cmap = discrete_cmap(13, 'Reds')
v = 6e-10
av_power_vib_effet.plot_topomap(fmin=8,fmax=30,tmin=0,tmax=2,vmin=0,vmax=v,cmap=my_cmap,colorbar=True)
v = 1.4e-9
av_power_vib_effet.plot_topomap(fmin=3,fmax=7,tmin=0,tmax=2,vmin=0,vmax=v,cmap=my_cmap,colorbar=True)

vmin = 0
vmax = 1e-9
av_power_vib_effet.plot(picks=["C3"],tmin=-1,tmax=3,vmin=vmin,vmax=vmax,cmap=my_cmap,colorbar=True)
av_power_vib_effet.plot(picks=["FC1"],tmin=-1,tmax=3,vmin=vmin,vmax=vmax,cmap=my_cmap,colorbar=True)

dureePreBaseline = 1 #3
dureePreBaseline = - dureePreBaseline
dureeBaseline = 1.0 #2.0
valeurPostBaseline = dureePreBaseline + dureeBaseline

baseline = (dureePreBaseline, valeurPostBaseline)
for tfr in liste_power_sujets:
    tfr.apply_baseline(baseline=baseline, mode='logratio', verbose=None)
      
    
av_power_vib_bl = mne.grand_average(liste_power_sujets,interpolate_bads=True)
av_power_vib_bl.save("../AV_TFR/all_sujets/effetVibSeule_BL-tfr.h5",overwrite=True)

my_cmap = discrete_cmap(13, 'RdBu')
my_cmap_rev = my_cmap.reversed()
vmax = 0.08
vmin = -vmax
av_power_vib_bl.plot_topomap(fmin=8,fmax=30,tmin=0,tmax=2,vlim=(vmin,vmax),cmap=my_cmap_rev,colorbar=True)
av_power_vib_bl.plot_topomap(fmin=3,fmax=7,tmin=0,tmax=2,vlim=(vmin,vmax),cmap=my_cmap_rev,colorbar=True)


vmax = 0.2
vmin = -vmax
av_power_vib_bl.plot(picks=["C3"],tmin=-1,tmax=3,vmin=vmin,vmax=vmax,cmap=my_cmap_rev,colorbar=True)
av_power_vib_bl.plot(picks=["FC1"],tmin=-1,tmax=3,vmin=vmin,vmax=vmax,cmap=my_cmap_rev,colorbar=True)


#===== POUR CONTROLE : premieres secondes apparition du pendule

av_power_controle = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion-tfr.h5")[0]
#  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule-tfr.h5")[0]

vmax = 0.08
vmin = -vmax

av_power_controle.plot_topomap(fmin=8,fmax=30,tmin=0,tmax=2,vlim=(vmin,vmax),cmap=my_cmap_rev,colorbar=True)
av_power_controle.plot_topomap(fmin=3,fmax=7,tmin=0,tmax=2,vlim=(vmin,vmax),cmap=my_cmap_rev,colorbar=True)


vmax = 0.2
vmin = -vmax
av_power_controle.plot(picks=["C3"],tmin=-1,tmax=3,vmin=vmin,vmax=vmax,cmap=my_cmap_rev,colorbar=True,fmin=0,fmax=40)
av_power_controle.plot(picks=["FC1"],tmin=-1,tmax=3,vmin=vmin,vmax=vmax,cmap=my_cmap_rev,colorbar=True,fmin=0,fmax=35)

av_power_controle_noBL = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion_noBaseline-tfr.h5")[0]
av_power_controle_noBL_pendule = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule_noBaseline-tfr.h5")[0]
av_power_controle_noBL_main = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main_noBaseline-tfr.h5")[0]

my_cmap = discrete_cmap(13, 'Reds')
v = 6e-10
av_power_controle_noBL.plot_topomap(fmin=8,fmax=30,tmin=0,tmax=2,vmin=0,vmax=v,cmap=my_cmap,colorbar=True)
v = 1.4e-9
av_power_controle_noBL.plot_topomap(fmin=3,fmax=7,tmin=0,tmax=2,vmin=0,vmax=v,cmap=my_cmap,colorbar=True)

vmin = 0
vmax = 1e-9
av_power_controle_noBL.plot(picks=["C3"],tmin=-1,tmax=3,vmin=vmin,vmax=vmax,cmap=my_cmap,colorbar=True,fmin=0,fmax=40)
av_power_controle_noBL.plot(picks=["FC1"],tmin=-1,tmax=3,vmin=vmin,vmax=vmax,cmap=my_cmap,colorbar=True,fmin=0,fmax=40)


#==============si on veut afficher le theta au fil du temps==========================

elec = 7 #FC1
elec = 11 #C3
data_vib_3_7_fc1 = np.mean(av_power_vib.data[elec][0:5],axis=0)
data_mainvib_3_7_fc1 = np.mean(av_power_controle_noBL.data[elec][0:5],axis=0)
data_mainvib_3_7_fc1 = data_mainvib_3_7_fc1[750:2250]
data_pendule_3_7_fc1 = np.mean(av_power_controle_noBL_pendule.data[elec][0:5],axis=0)
data_pendule_3_7_fc1 = data_pendule_3_7_fc1[750:2250]
data_effet_vib_3_7_fc1 = np.mean(av_power_vib_effet.data[elec][0:5],axis=0)


plt.plot(av_power_vib.times,data_vib_3_7_fc1,label='vib en nf')
plt.plot(av_power_vib.times,data_mainvib_3_7_fc1,label='main en nf de mainvib')#times un peu sale mais bon (on pourrait crop av_power_control)
plt.plot(av_power_vib.times,data_pendule_3_7_fc1,label='pendule en nf')
plt.plot(av_power_vib.times,data_effet_vib_3_7_fc1,label='vib hors nf')
plt.legend()
plt.show()


#essais de NF sur FC1

data_mainvib_3_7_fc1 = np.mean(av_power_controle_noBL.data[elec][0:5],axis=0)[750:7626]#-2 tmin
data_pendule_3_7_fc1 = np.mean(av_power_controle_noBL_pendule.data[elec][0:5],axis=0)[750:7626]
data_main_3_7_fc1 = np.mean(av_power_controle_noBL_main.data[elec][0:5],axis=0)[750:7626]

av_power_controle_noBL_pendule.crop(tmin = -2,tmax= 25.5)
av_power_controle_noBL_main.crop(tmin = -2,tmax= 25.5)
av_power_controle_noBL.crop(tmin = -2,tmax= 25.5)

plt.plot(av_power_controle_noBL_pendule.times,data_pendule_3_7_fc1,label='pendule en NF')
plt.plot(av_power_controle_noBL.times,data_mainvib_3_7_fc1,label='mainvib en nf')
plt.plot(av_power_controle_noBL_main.times,data_main_3_7_fc1,label='main en nf')
plt.legend()
plt.show()
raw_signal.plot(block=True)


#moving average
def computeMovingAverage(C3values,nvalues):
    arr_C3_movAverage = np.empty(nvalues, dtype=object) 
    compteur_moyenne = 1
    for i in range(1,nvalues):
        print("n value"+str(i))
        if compteur_moyenne == 5:
            print("continue")
            compteur_moyenne += 1
            continue#passe l'instance de la boucle
        elif compteur_moyenne == 6:
            compteur_moyenne = 1
            print("continue")
            continue
        offset = 125*i
        point_1 = C3values[250*i :250*(i+1) ].mean()
        point_2 = C3values[63+(250*i):63 + (250*(i+1))].mean()
        point_3 = C3values[125+(250*i) :125 +(250*(i+1))].mean()
        point_4 = C3values[188+(250*i) :188 +(250*(i+1)) ].mean()
        pointMoyenne = (point_1+point_2+point_3 + point_4)/4
        arr_C3_movAverage[i] = pointMoyenne
        compteur_moyenne += 1
    return arr_C3_movAverage

yo = computeMovingAverage(data_main_3_7_fc1,28) 
val_main = [val for val in yo if val is not None]
yo = computeMovingAverage(data_mainvib_3_7_fc1,28) 
val_mainVib = [val for val in yo if val is not None]
yo = computeMovingAverage(data_pendule_3_7_fc1,28) 
val_pendule = [val for val in yo if val is not None]
plt.plot(val_main,label='main en NF')
plt.plot(val_mainVib,label='mainVib en NF')
plt.plot(val_pendule,label='pendule en NF')
plt.legend()
plt.show()
raw_signal.plot(block=True)


#en moins pourri 
def compute_moving_average(arr, window_size, overlap):
    result = []
    step_size = window_size - overlap

    for i in range(0, len(arr) - window_size + 1, step_size):
        window = arr[i:i + window_size]
        average = np.mean(window)
        result.append(average)

    return np.array(result)


window_size = 75
overlap = 20

moving_average_main = compute_moving_average(data_main_3_7_fc1, window_size, overlap)
plt.plot(moving_average_main,label='main en NF')
moving_average_mainvib = compute_moving_average(data_mainvib_3_7_fc1, window_size, overlap)
plt.plot(moving_average_mainvib,label='mainvib en NF')
moving_average_pendule = compute_moving_average(data_pendule_3_7_fc1, window_size, overlap)
plt.plot(moving_average_pendule,label='pendule en NF')
plt.legend()
plt.show()
raw_signal.plot(block=True)