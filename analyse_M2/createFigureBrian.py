# -*- coding: utf-8 -*-
"""
Created on Wed May 25 17:17:34 2022

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

 

from mne import EvokedArray

#liste_tfr = load_tfr_data(liste_rawPathMain,"")

s08_main = load_tfr_data_windows(liste_rawPathMain[6:7],"",True)[0]
s24_pendule = load_tfr_data_windows(liste_rawPathPendule[22:23],"",True)[0]
s24_main = load_tfr_data_windows(liste_rawPathMain[22:23],"",True)[0]
s15_main = load_tfr_data_windows(liste_rawPathMain[13:14],"",True)[0]
s23_pendule = load_tfr_data_windows(liste_rawPathPendule[21:22],"",True)[0]

s11_main = load_tfr_data_windows(liste_rawPathMain[9:10],"",True)[0]

s24_pendule = load_tfr_data_windows(liste_rawPathPendule[22:23],"",True)[0]

data = s24_pendule
#conversion
#filtrer entre 8 et 30 Hz puis moyenner dessus
data.apply_baseline(baseline=(-3,-1),mode="logratio")
data_copy = data.copy()
data_copy.crop(fmin=8,fmax=30)

data_8_30 = np.mean(data_copy.data,axis=1)

epo = EvokedArray(data_8_30, data.info)

times = np.arange(11, 13,0.7)
fig,anim = epo.animate_topomap(
    times=times, ch_type='eeg', frame_rate=3, time_unit='s', blit=False,show=False)
raw_signal.plot(block=True)

anim.save('animation.gif', writer='imagemagick', fps=3)#s24 pendule

raw_signal.plot(block=True)

s22_pendule = load_tfr_data_windows(liste_rawPathPendule[20:21],"",True)[0]
data = s22_pendule
data.apply_baseline(baseline=(-3,-1),mode="logratio")
data_copy = data.copy()
data_copy.crop(fmin=8,fmax=30)

data_8_30 = np.mean(data_copy.data,axis=1)

epo = EvokedArray(data_8_30, data.info)

times = np.arange(6,24,1)
fig,anim = epo.animate_topomap(
    times=times, ch_type='eeg', frame_rate=3, time_unit='s', blit=False,show=False)
raw_signal.plot(block=True)

anim.save('animation_2.gif', writer='imagemagick', fps=3)#s011 main

import mne
from mne import EvokedArray
#en utilisant l'average map
av_power_main =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main-tfr.h5")[0]
av_power_mainIllusion =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion-tfr.h5")[0]
av_power_pendule =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule-tfr.h5")[0]


def animate_topo(data,doBaseline,tmin,tmax,step,fmin,fmax,vmin,vmax):
    if doBaseline:
        data.apply_baseline(baseline=(-3,-1),mode="logratio")
    data_copy = data.copy()
    data_copy.crop(fmin=fmin,fmax=fmax)
    
    data_8_30 = np.mean(data_copy.data,axis=1)
    
    epo = mne.EvokedArray(data_8_30, data.info)
    times = np.arange(tmin,tmax,step)
    fig,anim = epo.animate_topomap(
        times=times, ch_type='eeg', frame_rate=3, time_unit='s', blit=False,show=False,vmin=vmin,vmax=vmax)
    raw_signal.plot(block=True)
    return anim

anim = animate_topo(av_power_pendule,False,6,24,1,8,12,None,None)
anim.save('animation_pendule.gif', writer='imagemagick', fps=3)#s011 main

anim = animate_topo(av_power_main,False,6,24,1,8,30,None,None)
anim.save('animation_main.gif', writer='imagemagick', fps=3)#s011 main

anim = animate_topo(av_power_mainIllusion,False,6,24,1,8,30,None,None)
anim.save('animation_mainVibrations.gif', writer='imagemagick', fps=3)#s011 main



#find correct band 
list_data = [av_power_main,av_power_mainIllusion,av_power_pendule]
 
fullAverage = mne.grand_average(list_data)

fullAverage.plot(picks="C3",fmax=40)
raw_signal.plot(block=True)

v = 0.3
av_power_pendule.plot(picks="C3",fmin=3,fmax=50,vmax=v,vmin=-v)
av_power_main.plot(picks="C3",fmin=3,fmax=50,vmax=v,vmin=-v)
av_power_mainIllusion.plot(picks="C3",fmin=3,fmax=50,vmax=v,vmin=-v)
raw_signal.plot(block=True)

anim = animate_topo(av_power_pendule,False,0,24,1)
anim.save('../images/animation_pendule_full.gif', writer='imagemagick', fps=3)#s011 main

anim = animate_topo(av_power_mainIllusion,False,0,24,1)
anim.save('../images/animation_mainVibrations_full.gif', writer='imagemagick', fps=3)#s011 main

anim = animate_topo(av_power_main,False,0,24,1)
anim.save('../images/animation_main.gif', writer='imagemagick', fps=3)#s011 main

v = 0.25
av_power_pendule.plot(picks="C3",fmin=3,fmax=80,vmax=v,vmin=-v)
av_power_pendule.plot(picks=["FC1","CP5","CP1","CP5","C3"],fmin=3,fmax=80,vmax=v,vmin=-v,combine="mean")
av_power_main.plot(picks=["FC1","CP5","CP1","CP5","C3"],fmin=3,fmax=80,vmax=v,vmin=-v,combine="mean")
av_power_main.plot(picks="C3",fmin=3,fmax=80,vmax=v,vmin=-v)
av_power_mainIllusion.plot(picks=["FC1","CP5","CP1","CP5","C3"],fmin=3,fmax=80,vmax=v,vmin=-v,combine="mean")
av_power_mainIllusion.plot(picks="C3",fmin=3,fmax=80,vmax=v,vmin=-v)
raw_signal.plot(block=True)


# epoch_test = mne.Epochs(av_power_pendule,None)
# epoch_test.plot_psd(fmin=3,fmax=85,tmin=2,tmax=26.5)
# raw_signal.plot(block=True)

f, ax = plt.subplots()
ax.plot(freqs, av_power_pendule, color='k')
ax.set(title='Multitaper PSD (gradiometers)', xlabel='Frequency (Hz)',
       ylabel='Power Spectral Density (dB)')
plt.show()

