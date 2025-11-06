#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 12:48:49 2021

@author: claire.dussard
"""
import os 
import seaborn as sns
import pathlib
from handleData_subject import *
from functions.load_savedData import *
from functions.preprocessData_eogRefait import *
import numpy as np 


essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()


#create liste of file paths
essaisObsExecIm,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData_obsExec()
def createListeCheminsSignaux_obsExec(essaisObsExecIm,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates):
    #ex : pour tous essais mainSeule
    liste_rawPathObsExec = []
    listeEssais = essaisObsExecIm
    print(allSujetsDispo)
    for num_sujet in allSujetsDispo:
        print("sujet n° "+str(num_sujet))
        print("nom essai"+listeEssais[num_sujet])    
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
        sample_data_loc = nom_sujet+"/"+listeDatesFinale[num_sujet]+"/eeg"
        sample_data_dir = pathlib.Path(sample_data_loc)
        nom_essai = listeEssais[num_sujet].split(",")[0]#pour l'instant on prend le premier de chaque
        print(nom_essai)
        raw_path_sample = sample_data_dir/("BETAPARK_"+ date_nom_fichier + nom_essai+".vhdr")#il faudrait recup les epochs et les grouper ?
        liste_rawPathObsExec.append(raw_path_sample)
        print("APPENDED")
        
            
    print(len(liste_rawPathObsExec))#cb d'enregistrements recuperes
    return liste_rawPathObsExec

liste_rawPathObsExec = createListeCheminsSignaux_obsExec(essaisObsExecIm,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale,dates)

#pour se placer dans les donnees lustre
os.chdir("../../../../")
lustre_data_dir = "_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

eventObs = {'Observation mvt':16}
eventExec = {'Execution mvt':17}#attention c'est l'instruction, la video debute apres
eventImagine = {'Imagine mvt':18}
raw_signal = mne.io.read_raw_brainvision(liste_rawPathObsExec[4],preload=True,eog=('HEOG', 'VEOG'))
events = mne.events_from_annotations(raw_signal)


liste_epochsPreICA,liste_epochsSignal = pre_process_donnees_batch(liste_rawPathObsExec,1,0.1,90,[50,100],31,'Fz',[eventObs,eventExec,eventImagine],30)#que 2 premiers sujets
#[0] = obs, [1] = exec , [2] = imagine

liste_preICA = []
liste_signal = []
for suj in range(len(liste_epochsPreICA[0])):
    liste_epochs_sujet_preICA = [liste_epochsPreICA[0][suj],liste_epochsPreICA[1][suj],liste_epochsPreICA[2][suj]]
    epochsSujet_preICA = mne.concatenate_epochs(liste_epochs_sujet_preICA)
    liste_epochs_sujet_signal = [liste_epochsSignal[0][suj],liste_epochsSignal[1][suj],liste_epochsSignal[2][suj]]
    epochsSujet_signal = mne.concatenate_epochs(liste_epochs_sujet_signal)
    liste_preICA.append(epochsSujet_preICA)
    liste_signal.append(epochsSujet_signal)
    
listeICApreproc=[]
listeICA= []
for i in range(len(liste_preICA)):
    averageRefSignal_i,ICA_i = treat_indiv_data(liste_preICA[i],liste_signal[i],'Fz')
    listeICApreproc.append(averageRefSignal_i)
    listeICA.append(ICA_i)
     
    
saveEpochsAfterICA_avantdropBad_windows(listeICApreproc,liste_rawPathObsExec,True)
save_ICA_files(listeICA,liste_rawPathObsExec,True)

#reload epochs
listeICApreproc = load_data_postICA_preDropbad(liste_rawPathObsExec,"",True)

montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
for epochs in listeICApreproc:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)
        
        

EpochObs = [ls_epoch[0] for ls_epoch in listeICApreproc]
EpochExec = [ls_epoch[1] for ls_epoch in listeICApreproc]   
EpochIm = [ls_epoch[2] for ls_epoch in listeICApreproc]           


EpochData= EpochIm
liste_power_obs = []
liste_power_exec = []
liste_power_im = []
liste_power_sujets  = liste_power_im

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
            
#save_tfr_data(liste_power_sujets,liste_rawPathObsExec,"obs",True)
#save_tfr_data(liste_power_sujets,liste_rawPathObsExec,"exec",True)
save_tfr_data(liste_power_sujets,liste_rawPathObsExec,"imagine",True)     

listeTfrAv_exec = load_tfr_data_windows(liste_rawPathObsExec,"exec",True)
listeTfrAv_imagine = load_tfr_data_windows(liste_rawPathObsExec,"imagine",True) 


#sans BL

gd_average_exec.plot(picks="Cz",cmap=my_cmap,fmax=40,vmin=0,vmax=0.3e-9)
gd_average_im.plot(picks="Cz",cmap=my_cmap,fmax=40,vmin=0,vmax=0.3e-9)

gd_average_exec.plot(picks="C3",cmap=my_cmap,fmax=40,vmin=0,vmax=0.3e-9)
gd_average_im.plot(picks="C3",cmap=my_cmap,fmax=40,vmin=0,vmax=0.3e-9)

gd_average_exec.plot(picks="C4",cmap=my_cmap,fmin=8,fmax=35,vmin=0,vmax=0.3e-9)
gd_average_im.plot(picks="C4",cmap=my_cmap,fmin=8,fmax=35,vmin=0,vmax=0.3e-9)
raw_signal.plot(block=True)



for tfr in liste_power_sujets:
    tfr.apply_baseline(baseline=(-4,-2), mode='logratio', verbose=None)   
    

    
gd_average_obs = mne.grand_average(liste_power_sujets)
gd_average_obs.save("../AV_TFR/all_sujets/obs-tfr.h5",overwrite=True)

gd_average_exec = mne.grand_average(listeTfrAv_exec)
gd_average_exec.save("../AV_TFR/all_sujets/exec-tfr.h5",overwrite=True)
gd_average_exec.save("../AV_TFR/all_sujets/exec_noBaseline-tfr.h5",overwrite=True)

gd_average_im = mne.grand_average(listeTfrAv_imagine)
gd_average_im.save("../AV_TFR/all_sujets/imagine-tfr.h5",overwrite=True)

gd_average_obs =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/obs-tfr.h5")[0]
gd_average_exec =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/exec-tfr.h5")[0]
gd_average_im =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/imagine-tfr.h5")[0]

my_cmap = discrete_cmap(13, 'RdBu_r')
gd_average_obs.plot(picks="C3",cmap=my_cmap,fmax=80,vmin=-0.6,vmax=0.6)
gd_average_exec.plot(picks="C3",cmap=my_cmap,fmax=80,vmin=-0.6,vmax=0.6)
gd_average_im.plot(picks="C3",cmap=my_cmap,fmax=80,vmin=-0.6,vmax=0.6)
raw_signal.plot(block=True)

my_cmap = discrete_cmap(13, 'RdBu_r')
gd_average_obs.plot(picks="C4",cmap=my_cmap,fmax=80,vmin=-0.6,vmax=0.6)
gd_average_exec.plot(picks="C4",cmap=my_cmap,fmax=80,vmin=-0.6,vmax=0.6)
gd_average_im.plot(picks="C4",cmap=my_cmap,fmax=80,vmin=-0.6,vmax=0.6)
raw_signal.plot(block=True)

v = 0.45
gd_average_obs.plot_topomap(tmin=4,tmax=28,fmin=8,fmax=30,cmap=my_cmap,vmin=-v,vmax=v)
gd_average_exec.plot_topomap(tmin=4,tmax=28,fmin=8,fmax=30,cmap=my_cmap,vmin=-v,vmax=v)
gd_average_im.plot_topomap(tmin=4,tmax=28,fmin=8,fmax=30,cmap=my_cmap,vmin=-v,vmax=v)
raw_signal.plot(block=True)

for suj in liste_power_sujets[10:]:
    suj.plot_topomap(tmin=4,tmax=28,fmin=8,fmax=30,cmap=my_cmap,vmin=-v,vmax=v)
raw_signal.plot(block=True)   

for ep in EpochObs[0:10]:
    ep.plot()
raw_signal.plot(block=True)  




fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey='row')
# Supprimer les emplacements inutilisés
fig.delaxes(axs[0, 0])  # Colonne de gauche en haut
fig.delaxes(axs[0, 2])  # Colonne de droite en haut
vmax = 0.4
sensors=True

gd_average_im.plot_topomap(fmin=13,fmax=30,tmin=2.6,tmax=24,vlim=(-vmax,vmax),cmap = my_cmap,axes=axs[0,1],title="13-30Hz",sensors=sensors,cbar_fmt='%0.2f')

gd_average_im.plot_topomap(fmin=8,fmax=12,tmin=2.6,tmax=24,vlim=(-vmax,vmax),cmap = my_cmap,axes=axs[1,0],colorbar=False,title="8-12Hz",sensors=sensors)
gd_average_im.plot_topomap(fmin=13,fmax=20,tmin=2.6,tmax=24,vlim=(-vmax,vmax),cmap = my_cmap,axes=axs[1,1],colorbar=False,title="13-20Hz",sensors=sensors)
gd_average_im.plot_topomap(fmin=21,fmax=30,tmin=2.6,tmax=24,vlim=(-vmax,vmax),cmap = my_cmap,axes=axs[1,2],title="21-30Hz",sensors=sensors,cbar_fmt='%0.2f')

plt.tight_layout()
plt.show()


fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey='row')
vmax = 0.45
fig.delaxes(axs[0, 0])  # Colonne de gauche en haut
fig.delaxes(axs[0, 2])  # Colonne de droite en haut
sensors =True
gd_average_exec.plot_topomap(fmin=13,fmax=30,tmin=2.6,tmax=24,vlim=(-vmax,vmax),cmap = my_cmap,axes=axs[0,1],title="13-30Hz",sensors=sensors,cbar_fmt='%0.2f')
gd_average_exec.plot_topomap(fmin=8,fmax=12,tmin=2.6,tmax=24,vlim=(-vmax,vmax),cmap = my_cmap,axes=axs[1,0],colorbar=False,title="8-12Hz",sensors=sensors)
gd_average_exec.plot_topomap(fmin=13,fmax=20,tmin=2.6,tmax=24,vlim=(-vmax,vmax),cmap = my_cmap,axes=axs[1,1],colorbar=False,title="13-20Hz",sensors=sensors)
gd_average_exec.plot_topomap(fmin=21,fmax=30,tmin=2.6,tmax=24,vlim=(-vmax,vmax),cmap = my_cmap,axes=axs[1,2],title="21-30Hz",sensors=sensors,cbar_fmt='%0.2f')

plt.tight_layout()
plt.show()


#================================
liste_epochsPreICA,liste_epochsSignal = pre_process_donnees(liste_rawPathObsExec,1,0.1,90,[50,100],31,'Fz',eventExec,30)#que 2 premiers sujets

listeICApreproc=[]
listeICA= []
for i in range(len(liste_epochsPreICA)):
    averageRefSignal_i,ICA_i = treat_indiv_data(liste_epochsPreICA[i],liste_epochsSignal[i],'Fz')
    listeICApreproc.append(averageRefSignal_i)
    listeICA.append(ICA_i)
 

montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
for epochs in listeICApreproc:
    if epochs!=None:
        epochs.set_montage(montageEasyCap)
        
EpochData= listeICApreproc
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
    

        
for tfr in liste_power_sujets:
    tfr.apply_baseline(baseline=(-4,-2), mode='logratio', verbose=None)
    
gd_average = mne.grand_average(liste_power_sujets)
my_cmap = discrete_cmap(13, 'RdBu_r')
gd_average.plot(picks="C3",cmap=my_cmap,fmax=40,vmin=-0.6,vmax=0.6)
gd_average.plot_topomap(tmin=4,tmax=28,fmin=8,fmax=30,cmap=my_cmap,vmin=-0.4,vmax=0.4)
raw_signal.plot(block=True)
    
  # =============================================================================================================
#on reprend la fct, j'y comprends rien #en fait on avait un enregistrement par condition et la tout est mixe :)
#16 = obs, 17 = exec, 18 = IM obs #pb pck 13 = debut affichage video mais jamais capte # en fait 13 est remplace par 19 par le code python :')
#en fait on a qu'a utiliser les 16 17 et 18 mais decaler le debut de l'epoch de 3s #en fait on peut pas le faire, tmin est forcement positif #on peut pas passer en entree juste l'evenement qu'on veut pck il a pas la bonne taille 
#on peut epocher tous les 19 et ensuite les trier en indexant  type epochs[:10])  
for i in range(len(listeICApreproc)+1):#a quoi sert la double boucle for ? 
    liste_tous_epochs = []
    for numSignal in range(len(listeICApreproc)):#a quoi sert la double boucle for ? 
        events = mne.events_from_annotations(listeICApreproc[numSignal])
        #recuperer les temps pour modifier les numeros d'events
        #corriger les events ID pour les rendre uniques 
        indicesDebutsVideo = np.where(events[0][:,2] ==19)[0]
        tpsDebutObs= events[0][indicesDebutsVideo[0]][0]
        tpsDebutExec= events[0][indicesDebutsVideo[1]][0]
        tpsDebutMI= events[0][indicesDebutsVideo[2]][0]
        signal = listeICApreproc[numSignal]
        eventsCorriges = np.array([
            [tpsDebutObs, 0 , 16],
            [tpsDebutExec,0,17],
            [tpsDebutMI,0,18]])
        event_id={'Observation mvt':16,
                  'Execution mvt': 17,
                  'Imagination mvt': 18}     #ils etaient cropped a -0.2 ?    
        epochsCibles = mne.Epochs(signal,eventsCorriges,event_id,tmin=-5.0,tmax = 29.0,baseline=(-5, 0),picks= ["C3", "Cz", "C4"])
        liste_tous_epochs.append(epochsCibles)


tousEpochs = mne.epochs.concatenate_epochs(liste_tous_epochs)
#save epochs
tousEpochs.save("MI_exec_obs_baseline5s_epo.fif") 
tousEpochs = mne.read_epochs("MI_exec_obs_baseline5s_epo.fif")#save point :)
#ne save pas les channels ?

#VERSION UNE SEULE CONDITION
# from mne.time_frequency import tfr_multitaper
# from mne.viz.utils import center_cmap
# import matplotlib.pyplot as plt
# #https://mne.tools/stable/auto_examples/time_frequency/time_frequency_erds.html

# # epoch data ##################################################################
# tmin, tmax = -5, 29  # define epochs around events (in s)
# #event_ids = dict(Affichage_video=13)  # map event IDs to tasks
# #event_ids = {'19': 19}#dict(Affichage_vidéo=19)
# epochs = tousEpochs #A MODIFIER

# # compute ERDS maps ###########################################################
# freqs = np.arange(2, 36, 1)  # frequencies from 2-35Hz
# n_cycles = freqs  # use constant t/f resolution
# vmin, vmax = -1.5, 1.5  # set min and max ERDS values in plot
# baseline = [-5.0, 0]  # baseline interval (in s)
# #preferable d'avoir condition average baseline
# cmap = center_cmap(plt.cm.RdBu, vmin, vmax)  # zero maps to white (https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html)
# kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
#               buffer_size=None, out_type='mask')  # for cluster test

# # Run TF decomposition overall epochs
# tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles = n_cycles,
#                      use_fft=True, return_itc=False, average=False,
#                      decim=2)
# tfr = tfr.crop(tmin, tmax)
# tfr = tfr.apply_baseline(baseline, mode="percent")#baseline
# for event in event_ids:
#     # select desired epochs for visualization
#     tfr_ev = tfr[event]
#     fig, axes = plt.subplots(1, 4, figsize=(12, 4),
#                              gridspec_kw={"width_ratios": [10, 10, 10, 1]})
#     for ch, ax in enumerate(axes[:-1]):  # for each channel
#         # plot TFR (ERDS map with masking)
#         tfr_ev.average().plot([ch], vmin=vmin, vmax=vmax, cmap=(cmap, False),
#                               axes=ax, colorbar=False, show=False)#, mask=mask,
#                               #mask_style="mask")

#         ax.set_title(epochs.ch_names[ch], fontsize=10)
#         ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
#         if ch != 0:
#             ax.set_ylabel("")
#             ax.set_yticklabels("")
#     fig.colorbar(axes[0].images[-1], cax=axes[-1])
#     fig.suptitle("ERDS ({})".format(event))
#     fig.show(block=True)
#     fig.savefig("5sbaseline_MISeul.png")


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# # ERD as function of time
# df = tfr.to_data_frame(time_format=None, long_format=False) #cette fonction n'existe que depuis version 0.23.0
# #module load MNE/0.23.0
# print(df.condition.value_counts())#verification qu'on a toutes les conditions
# df.to_csv("data15Sujets_condition.csv")
# df.to_csv("data15Sujets_condition_baselineModif.csv")
# df = pd.read_csv("data15Sujets_condition_baselineModif.csv")#resume session shortcut :)
# %matplotlib inline
# # Map to frequency bands:
# freq_bounds = {'_': 0,
#                'delta': 3,
#                'theta': 8,
#                'alpha': 13,
#                'beta': 30,
#                'gamma': 140}
# df['band'] = pd.cut(df['freq'], list(freq_bounds.values()),#dans tfr on a defini un pas de frequence
#                     labels=list(freq_bounds)[1:])#, ici on regroupe les freqs par bande

# # Filter to retain only relevant frequency bands:
# freq_bands_of_interest = ['alpha','beta']
# df = df[df.band.isin(freq_bands_of_interest)]
# df['band'] = df['band'].cat.remove_unused_categories() #bad practice : tu peux lancer le code qu'une fois apres : Number of rows must be a positive integer, not 0
# want_chs = ['C3', 'Cz', 'C4']#solution = reload df
# # Order channels for plotting
# df['channel']=df['channel'].astype('category') #fixed ? ou casse tout ?
# #df['channel'] = df['channel'].cat.reorder_categories(want_chs, ordered=True) #bug ?


# g = sns.FacetGrid(df, row='band', col='channel', margin_titles=True)
# yo = g.map(sns.lineplot, 'time', 'value', 'condition', n_boot=10)
# axline_kw = dict(color='black', linestyle='dashed', linewidth=0.5, alpha=0.5)
# yo = g.map(plt.axhline, y=0, **axline_kw)
# yo = g.map(plt.axvline, x=0, **axline_kw)
# yo = g.set(ylim=(None, 10.0))
# yo = g.set_axis_labels("Time (s)", "ERDS (%)")
# yo = g.set_titles(col_template="{col_name}", row_template="{row_name}")
# yo = g.add_legend(ncol=2, loc='lower center')
# yo = g.fig.subplots_adjust(left=0.3, right=1.9, top=1.9, bottom=0.2)


# #3 eme graph

# df_mean = (df.query('time > 1')
#              .groupby(['condition', 'epoch', 'band', 'channel'])[['value']]
#              .mean()
#              .reset_index())

# g = sns.FacetGrid(df_mean, col='condition',
#                   margin_titles=True)
# g = (g.map(sns.violinplot, 'channel', 'value', 'band', n_boot=10,
#            palette='deep', order=['C3', 'Cz', 'C4'],
#            hue_order=freq_bands_of_interest,
#            linewidth=0.5)
#       .add_legend(ncol=4, loc='lower center'))

# yo = g.map(plt.axhline, **axline_kw)
# yo = g.set_axis_labels("", "ERDS (%)")
# yo = g.set_titles(col_template="{col_name}", row_template="{row_name}")
# yo = g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)

# event_ids={'Observation mvt':16,
#                   'Execution mvt': 17,
#                   'Imagination mvt': 18} 
# # 1 er graph

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# import mne
# from mne.datasets import eegbci
# from mne.io import concatenate_raws, read_raw_edf
# from mne.time_frequency import tfr_multitaper
# from mne.stats import permutation_cluster_1samp_test as pcluster_test
# from mne.viz.utils import center_cmap
# vmin, vmax = -1.5, 1.5 
# epochs = tousEpochs
# for event in event_ids:
#     # select desired epochs for visualization
#     tfr_ev = tfr[event]
#     fig, axes = plt.subplots(1, 4, figsize=(12, 4),
#                              gridspec_kw={"width_ratios": [10, 10, 10, 1]})
#     for ch, ax in enumerate(axes[:-1]):  # for each channel
#         # positive clusters
#         _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=1, **kwargs)
#         # negative clusters
#         _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=-1,
#                                      **kwargs)

#         # note that we keep clusters with p <= 0.05 from the combined clusters
#         # of two independent tests; in this example, we do not correct for
#         # these two comparisons
#         c = np.stack(c1 + c2, axis=2)  # combined clusters
#         p = np.concatenate((p1, p2))  # combined p-values
#         mask = c[..., p <= 0.05].any(axis=-1)

#         # plot TFR (ERDS map with masking)
#         tfr_ev.average().plot([ch], vmin=vmin, vmax=vmax, cmap=(cmap, False),
#                               axes=ax, colorbar=False, show=False, mask=mask,
#                               mask_style="mask")

#         ax.set_title(epochs.ch_names[ch], fontsize=10)
#         ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
#         if ch != 0:
#             ax.set_ylabel("")
#             ax.set_yticklabels("")
#     fig.colorbar(axes[0].images[-1], cax=axes[-1])
#     fig.suptitle("ERDS ({})".format(event))
#     fig.show()
    
# #sortir des cartes EEG topo
# #refaire les epochs mais sans les picks