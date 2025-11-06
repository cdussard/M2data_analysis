#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 15:50:06 2021

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

nom_essai = "2-2"

SujetsPbNomFichiers = []
SujetsExclusAnalyse = [1,4]
#separer l'analyse en sujets ayant reussi et rate ?
#voir si ça se voyait a l'IM seule
SujetsAvecPb = np.unique(SujetsPbNomFichiers + SujetsExclusAnalyse)
allSujetsDispo = [i for i in range(nombreTotalSujets+1)]
for sujet_pbmatique in SujetsAvecPb:  
    allSujetsDispo.remove(sujet_pbmatique)
print(len(allSujetsDispo))

#on commence au 3 (reste pbs noms)
liste_rawPath = []
for num_sujet in allSujetsDispo:
    print("sujet n° "+str(num_sujet))
    nom_sujet = listeNumSujetsFinale[num_sujet]
    date_nom_fichier = dates[num_sujet][-4:]+"-"+dates[num_sujet][3:5]+"-"+dates[num_sujet][0:2]+"_"
    dateSession = listeDatesFinale[num_sujet]
    sample_data_loc = listeNumSujetsFinale[num_sujet]+"/"+listeDatesFinale[num_sujet]+"/eeg"
    sample_data_dir = pathlib.Path(sample_data_loc)
    raw_path_sample = sample_data_dir/("BETAPARK_"+ date_nom_fichier + nom_essai+".vhdr")#il faudrait recup les epochs et les grouper ?
    liste_rawPath.append(raw_path_sample)

print(liste_rawPath)

listeAverageRef,listeRaw = pre_process_donnees(liste_rawPath[12:14],1,80,[50,100],31,'Fz',False,[])
listeICApreproc,listeICA = ICA_preproc(listeAverageRef,listeRaw,[],31,98,False)

#rejection criteria
reject_criteria = dict(mag=3000e-15,     # 3000 fT
                       grad=3000e-13,    # 3000 fT/cm
                       eeg=100e-6,       # 100 µV
                       eog=200e-6)      

event_id={'Imagination seule':21}
events = mne.events_from_annotations(listeICApreproc[0])
for i in range(len(listeICApreproc)+1):
    liste_tous_epochs = []
    for numSignal in range(len(listeICApreproc)):
        events = mne.events_from_annotations(listeICApreproc[numSignal])[0]#baseline=(-2, 0)
        signal = listeICApreproc[numSignal]
        epochsCibles = mne.Epochs(signal,events,event_id,tmin=-5.0,tmax = 30.0,reject=reject_criteria,baseline=None,picks= ["C3", "Cz", "C4"])
        liste_tous_epochs.append(epochsCibles)
 
tousEpochs.plot(block=True)#verif visuelle voire exclusion si besoin

tousEpochs = mne.epochs.concatenate_epochs(liste_tous_epochs)#15 * 2 epochs
tousEpochs.save("MI_seul_baseline3s_15suj_epo.fif") 
tousEpochs = mne.read_epochs("MI_seul_baseline3s_15suj_epo.fif") #loading point 

#VERSION UNE SEULE CONDITION
from mne.time_frequency import tfr_multitaper
from mne.viz.utils import center_cmap
import matplotlib.pyplot as plt
#https://mne.tools/stable/auto_examples/time_frequency/time_frequency_erds.html

# epoch data ##################################################################
tmin, tmax = -5, 30  # define epochs around events (in s)
#event_ids = dict(Affichage_video=13)  # map event IDs to tasks
#event_ids = {'19': 19}#dict(Affichage_vidéo=19)
epochs = tousEpochs #A MODIFIER

# compute ERDS maps ###########################################################
freqs = np.arange(2, 36, 1)  # frequencies from 2-35Hz
n_cycles = freqs  # use constant t/f resolution
vmin, vmax = -1.5, 1.5  # set min and max ERDS values in plot
baseline = [-3.0, 0]  # baseline interval (in s) #muted par la baseline ds les epochs
#preferable d'avoir condition average baseline
cmap = center_cmap(plt.cm.RdBu, vmin, vmax)  # zero maps to white (https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html)
kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
              buffer_size=None, out_type='mask')  # for cluster test

# Run TF decomposition overall epochs
tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles = n_cycles,
                     use_fft=True, return_itc=False, average=False,
                     decim=2)
#tfr = tfr.crop(tmin, tmax)#est ce qu'il faudrait pas cropper apres ? essayer l'un puis l'autre et l'inverse
tfr = tfr.apply_baseline(baseline, mode="percent")#baseline mean &log ratio resultats bizarres

df = tfr.to_data_frame(time_format=None, long_format=True)
#df.to_csv("data15Sujets_IMseule_long.csv")

event_ids={'Imagination seule':21}
for event in event_ids:
    # select desired epochs for visualization
    tfr_ev = tfr[event]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4),
                             gridspec_kw={"width_ratios": [10, 10, 10, 1]})
    for ch, ax in enumerate(axes[:-1]):  # for each channel
        # plot TFR (ERDS map with masking)
        tfr_ev.average().plot([ch], vmin=vmin, vmax=vmax, cmap=(cmap, False),
                              axes=ax, colorbar=False, show=False)#, mask=mask,
                              #mask_style="mask")

        ax.set_title(epochs.ch_names[ch], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
        if ch != 0:
            ax.set_ylabel("")
            ax.set_yticklabels("")
    yo = fig.colorbar(axes[0].images[-1], cax=axes[-1])
    yo = fig.suptitle("ERDS ({})".format(event))
    #fig.show()
    yo = fig.savefig("3sBLpercent_IMseule_15sujets.png")
    
#deuxieme graphique

#ne peut fonctionner que si on a une colonne channel dans le df (long_format = True)
df = tfr.to_data_frame(time_format=None, long_format=False) #pas d'info channel
#info channel
df = tfr.to_data_frame(time_format=None, long_format=True) 

freq_bounds = {'_': 0,
               'delta': 3,
               'theta': 8,
               'alpha': 13,
               'beta': 30,
               'gamma': 140}
df['band'] = pd.cut(df['freq'], list(freq_bounds.values()),#dans tfr on a defini un pas de frequence
                    labels=list(freq_bounds)[1:])#, ici on regroupe les freqs par bande

# Filter to retain only relevant frequency bands:
freq_bands_of_interest = ['alpha','beta']
df = df[df.band.isin(freq_bands_of_interest)]
df['band'] = df['band'].cat.remove_unused_categories() #bad practice : tu peux lancer le code qu'une fois apres : Number of rows must be a positive integer, not 0
want_chs = ['C3', 'Cz', 'C4']#solution = reload df
# Order channels for plotting
df['channel']=df['channel'].astype('category') #fixed ? ou casse tout ?
#df['channel'] = df['channel'].cat.reorder_categories(want_chs, ordered=True) #bug ?


g = sns.FacetGrid(df, row='band', col='channel', margin_titles=True)
yo = g.map(sns.lineplot, 'time', 'value', 'condition', n_boot=10)
axline_kw = dict(color='black', linestyle='dashed', linewidth=0.5, alpha=0.5)
yo = g.map(plt.axhline, y=0, **axline_kw)
yo = g.map(plt.axvline, x=0, **axline_kw)
yo = g.set(ylim=(None, 10.0))
yo = g.set_axis_labels("Time (s)", "ERDS (%)")
yo = g.set_titles(col_template="{col_name}", row_template="{row_name}")
yo = g.add_legend(ncol=2, loc='lower center')
yo = g.fig.subplots_adjust(left=0.3, right=1.9, top=1.9, bottom=0.2)

#3eme graph
df_mean = (df.query('time > 1')
             .groupby(['condition', 'epoch', 'band', 'channel'])[['value']]
             .mean()
             .reset_index())

g = sns.FacetGrid(df_mean, col='condition',
                  margin_titles=True)
g = (g.map(sns.violinplot, 'channel', 'value', 'band', n_boot=10,
           palette='deep', order=['C3', 'Cz', 'C4'],
           hue_order=freq_bands_of_interest,
           linewidth=0.5)
      .add_legend(ncol=4, loc='lower center'))

yo = g.map(plt.axhline, **axline_kw)
yo = g.set_axis_labels("", "ERDS (%)")
yo = g.set_titles(col_template="{col_name}", row_template="{row_name}")
yo = g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)


#power spectrum
from mne.minimum_norm import read_inverse_operator, compute_source_psd
bands = [(0, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
         (12, 35, 'Beta'), (30, 45, 'Gamma')]
montageEasyCap = mne.channels.make_standard_montage('easycap-M1')
montagedEpochs=tousEpochs.set_montage(montageEasyCap)
#montagedEpochs.plot_psd_topomap(bands = bands)#output etrange 3 elecs (normal : on a vire toutes les autres x)
montagedEpochs.plot_psd(1,50,estimate="power",average=True)#output etrange

#pour debloquer les graphs
raw_signal.plot(block=True)