#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:11:23 2021

@author: claire.dussard
"""
import os 
import seaborn as sns
import pathlib
#recuperation des donnees sujet IM seule
#necessite d'avoir execute handleData_subject.py avant 
import numpy as np 
#pour se placer dans les donnees lustre
os.chdir("../../../../..")
lustre_data_dir = "cenir/analyse/meeg/BETAPARK/_RAW_DATA"
lustre_path = pathlib.Path(lustre_data_dir)
os.chdir(lustre_path)

nom_essai = "4"
essaisFeedbackSeul = ["pas_enregistre","sujet jeté",
"4","4","sujet jeté","4","4","4","4","MISSING","4","4",
"4","4","4","4","4","4","4","4-b","4","4","4","4","4","4"]
sujetsPb = [0,9]
for sujetpb in sujetsPb:
    allSujetsDispo.remove(sujetpb)
# essaisFeedbackSeul = [nom_essai for i in range(25)]

    
liste_rawPathEffetFBseul = createListeCheminsSignaux(essaisFeedbackSeul,listeNumSujetsFinale, allSujetsDispo,SujetsPbNomFichiers,listeDatesFinale)

#ordre inverse au milieu de la manip pr eviter effets ordre)

event_id_mainIllusion = {'Main illusion seule': 26}
event_id_pendule={'Pendule seul':23}  
event_id_main={'Main seule': 24}  

nbSujets = 4
SujetsDejaTraites = 22
rawPathEffetFBseul_sujets = liste_rawPathEffetFBseul[SujetsDejaTraites:SujetsDejaTraites+nbSujets]

listeEpochs_main,listeICA,listeEpochs_pendule,listeEpochs_mainIllusion = all_conditions_analysis_FBseul(allSujetsDispo,rawPathEffetFBseul_sujets,
                            event_id_main,event_id_pendule,event_id_mainIllusion,
                            0.1,1,90,[50,100],'Fz')

saveEpochsAfterICA_FBseul(listeEpochs_main,rawPathEffetFBseul_sujets,"main")
saveEpochsAfterICA_FBseul(listeEpochs_pendule,rawPathEffetFBseul_sujets,"pendule")
saveEpochsAfterICA_FBseul(listeEpochs_mainIllusion,rawPathEffetFBseul_sujets,"mainIllusion")
save_ICA_files(listeICA,rawPathEffetFBseul_sujets)

#=============== EPOCHS POWER ==========================
EpochDataMain = load_data_postICA(liste_rawPathEffetFBseul,"main")

EpochDataPendule = load_data_postICA(liste_rawPathEffetFBseul,"pendule")

EpochDataMainIllusion = load_data_postICA(liste_rawPathEffetFBseul,"mainIllusion")

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

save_tfr_data(liste_power_mainIllusion,liste_rawPathEffetFBseul,"mainIllusion")

save_tfr_data(liste_power_sujets,liste_rawPathEffetFBseul,"main")

save_tfr_data(liste_power_sujets,liste_rawPathEffetFBseul,"pendule")

#============= compute average power ==============
dureePreBaseline = 4
dureePreBaseline = - dureePreBaseline
dureeBaseline = 3.0
valeurPostBaseline = dureePreBaseline + dureeBaseline

baseline = (dureePreBaseline, valeurPostBaseline)
for tfr in liste_power_sujets:
    tfr.apply_baseline(baseline=baseline, mode='logratio', verbose=None)

#================compute grand average===============================================
mode = 'logratio'
av_power_mainIllusion = mne.grand_average(liste_power_mainIllusion,interpolate_bads=True)
save_topo_data(av_power_mainIllusion,dureePreBaseline,valeurPostBaseline,"all_sujets",mode,"effet_mainIllusionSeule",False,1.0,24.0)#can be improved with boolean Params for alpha etcliste_tfr,interpolate_bads=True)

av_power_main = mne.grand_average(liste_power_sujets,interpolate_bads=True)
save_topo_data(av_power_main,dureePreBaseline,valeurPostBaseline,"all_sujets",mode,"effet_mainSeule",False,1.0,24.0)#can be improved with boolean Params for alpha etcliste_tfr,interpolate_bads=True)

av_power_pendule = mne.grand_average(liste_power_sujets,interpolate_bads=True)
save_topo_data(av_power_pendule,dureePreBaseline,valeurPostBaseline,"all_sujets",mode,"effet_penduleSeul",False,1.0,24.0)#can be improved with boolean Params for alpha etcliste_tfr,interpolate_bads=True)


avpower_main_moins_mainIllusion = av_power_main - av_power_mainIllusion

avpower_main_moins_pendule = av_power_main - av_power_pendule

save_topo_data(avpower_main_moins_pendule,dureePreBaseline,valeurPostBaseline,"all_sujets",mode,"effet_main-pendule",False,1.0,24.0)
save_topo_data(avpower_main_moins_mainIllusion,dureePreBaseline,valeurPostBaseline,"all_sujets",mode,"effet_main-mainIllusion",False,1.0,24.0)
#============================================================================================================
# SujetsPbNomFichiers = [0,1,2,3,4,5,6]#9 : manque recording (on a l'OV)
# SujetsExclusAnalyse = [1,4]
# #separer l'analyse en sujets ayant reussi et rate ?
# #voir si ça se voyait a l'IM seule
# SujetsAvecPb = np.unique(SujetsPbNomFichiers + SujetsExclusAnalyse)
# allSujetsDispo = [i for i in range(nombreTotalSujets+1)]
# for sujet_pbmatique in SujetsAvecPb:  
#     allSujetsDispo.remove(sujet_pbmatique)
# print(len(allSujetsDispo))

# #on commence au 3 (reste pbs noms)
# liste_rawPath = []
# for num_sujet in allSujetsDispo:
#     print("sujet n° "+str(num_sujet))
#     nom_sujet = listeNumSujetsFinale[num_sujet]
#     date_nom_fichier = dates[num_sujet][-4:]+"-"+dates[num_sujet][3:5]+"-"+dates[num_sujet][0:2]+"_"
#     dateSession = listeDatesFinale[num_sujet]
#     sample_data_loc = listeNumSujetsFinale[num_sujet]+"/"+listeDatesFinale[num_sujet]+"/eeg"
#     sample_data_dir = pathlib.Path(sample_data_loc)
#     raw_path_sample = sample_data_dir/("BETAPARK_"+ date_nom_fichier + nom_essai+".vhdr")
#     liste_rawPath.append(raw_path_sample)

# print(liste_rawPath)

# listeAverageRef,listeRaw = pre_process_donnees(liste_rawPath,1,80,[50,100],31,'Fz',False,[])
# listeICApreproc,listeICA = ICA_preproc(listeAverageRef,listeRaw,[],31,98,False)

# events = mne.events_from_annotations(listeICApreproc[1])
# #il faut retrouver les essais de chaque type
# #23 24 25 26 #ordre inverse au milieu de la manip pr eviter effets ordre)
# event_ids={'Pendule seul':23,
#                   'Main seule': 24,
#                   'Main tactile': 25,
#                   'Main illusion': 26} 

# liste_tous_epochs = []
# for numSignal in range(1,len(listeICApreproc)):
#     if numSignal ==9:
#         pass
#     else:
#         print("\n numero "+str(numSignal))
#         events = mne.events_from_annotations(listeICApreproc[numSignal])[0]#baseline=(-2, 0)
#         signal = listeICApreproc[numSignal]
#         epochsCibles = mne.Epochs(signal,events,event_ids,tmin=-5.0,tmax = 28.0,baseline=None,picks=["C3","Cz","C4"])
#         liste_tous_epochs.append(epochsCibles)


# tousEpochs.plot(block=True)#verif visuelle voire exclusion si besoin
# tousEpochs = mne.epochs.concatenate_epochs(liste_tous_epochs)#15 * 2 epochs
# tousEpochs.save("FB_seul_12suj_epo.fif") #loading point :)
# tousEpochs = mne.read_epochs("FB_seul_12suj_epo.fif")
# #sortir des cartes EEG topo


# tousEpochs.plot_psd_topomap(bands = bands)#output etrange 3 elecs (normal : on a vire toutes les autres x)
# tousEpochs.plot_psd(1,50,estimate="power",average=True)#output etrange


# #pour debloquer les graphs
# raw_signal.plot(block=True)

# from mne.time_frequency import tfr_multitaper
# from mne.viz.utils import center_cmap
# import matplotlib.pyplot as plt
# #1er graph
# epochs = tousEpochs
# # compute ERDS maps ###########################################################
# freqs = np.arange(2, 36, 3)  # frequencies from 2-35Hz
# n_cycles = freqs  # use constant t/f resolution
# vmin, vmax = -1.5, 1.5  # set min and max ERDS values in plot
# baseline = [-3.0, 0]  # baseline interval (in s) #muted par la baseline ds les epochs
# #preferable d'avoir condition average baseline
# cmap = center_cmap(plt.cm.RdBu, vmin, vmax)  # zero maps to white (https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html)
# kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
#               buffer_size=None, out_type='mask')  # for cluster test

# tmin = -1
# tmax = 30.0
# # Run TF decomposition overall epochs
# tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles = n_cycles,
#                      use_fft=True, return_itc=False, average=False,
#                      decim=2,picks=['C3','Cz','C4'])#ajouter les neighboring electrodes du laplacien ?
# tfr = tfr.crop(tmin, tmax)#est ce qu'il faudrait pas cropper apres ? essayer l'un puis l'autre et l'inverse
# tfr = tfr.apply_baseline(baseline, mode="percent")#baseline mean &log ratio resultats bizarres

# #df = tfr.to_data_frame(time_format=None, long_format=True)
# #df.to_csv("data15Sujets_IMseule_long.csv")

# event_ids={'Pendule seul':23,
#                   'Main seule': 24,
#                   'Main tactile': 25,
#                   'Main illusion': 26} 
# #carte temps frequence
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# import mne
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
#         print("channel"+str(tfr.ch_names[ch]))
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

#         ax.set_title(tfr.ch_names[ch], fontsize=10)#change epoch pr tfr pck picks effectue au tfr, pas au niveau epoch
#         ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
#         if ch != 0:
#             ax.set_ylabel("")
#             ax.set_yticklabels("")
#     fig.colorbar(axes[0].images[-1], cax=axes[-1])
#     fig.suptitle("ERDS ({})".format(event))
#     fig.show()
    
# #evolution au cours du temps
# df = tfr.to_data_frame(time_format=None, long_format=True)

# # Map to frequency bands:
# freq_bounds = {'_': 0,
#                'below': 8,
#                '8-30Hz': 30,
#                'above': 140
#                }
# df['band'] = pd.cut(df['freq'], list(freq_bounds.values()),
#                     labels=list(freq_bounds)[1:])

# # Filter to retain only relevant frequency bands:
# freq_bands_of_interest = ['8-30Hz', 'above', 'below']
# df = df[df.band.isin(freq_bands_of_interest)]
# df['band'] = df['band'].cat.remove_unused_categories()
# want_chs = ['C3', 'Cz', 'C4']
# # Order channels for plotting:
# df['channel'] = df['channel'].cat.reorder_categories(want_chs, ordered=True)

# g = sns.FacetGrid(df, row='band', col='channel', margin_titles=True)
# g.map(sns.lineplot, 'time', 'value', 'condition', n_boot=10)
# axline_kw = dict(color='black', linestyle='dashed', linewidth=0.5, alpha=0.5)
# g.map(plt.axhline, y=0, **axline_kw)
# g.map(plt.axvline, x=0, **axline_kw)
# g.set(ylim=(None, 1.5))
# g.set_axis_labels("Time (s)", "ERDS (%)")
# g.set_titles(col_template="{col_name}", row_template="{row_name}")
# g.add_legend(ncol=2, loc='lower center')
# g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)
    
#  #reconstruire l'electrode C3 laplacien ?
#  # Create a bipolar EOG channel.
# # The resulting signal should be EOG 061 - EOG 062, so all zeros, because in
# # this example we just copied the channel.
# #C3reconstructed = mne.set_bipolar_reference(raw, 'EOG 061', 'EOG 062', ch_name='HEOG') #pas bon, on a besoin de multiplier, ici c'est un - autre
# #plutot utiliser raw.get_data et travailler sur les tableaux surement
# #sortir les donnees au cours du temps avec openvibe ? (ici ça desynchronise mais est ce qu'avec le data processing openvibe les sujets etaient
# #vrmt avantages ?)