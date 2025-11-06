# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 13:01:13 2024

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


av_power_main =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main-tfr.h5")[0]
av_power_mainIllusion =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion-tfr.h5")[0]
av_power_pendule =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule-tfr.h5")[0]

av_power_mainP =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/effet_main-tfr.h5")[0]
av_power_mainIllusionP =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/effet_mainIllusion-tfr.h5")[0]
av_power_penduleP =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/effet_pendule-tfr.h5")[0]


#♥all TFR plots
fmax = 40
pick = "C3"
vmin = -0.5
vmax = -vmin
fig, axs = plt.subplots(2, 3, sharey=True)
colorbar = False
av_power_pendule.plot(picks=pick,axes=axs[0,0],fmax=fmax,vmin=vmin,vmax=vmax,colorbar=colorbar)
av_power_main.plot(picks=pick,axes=axs[0,1],fmax=fmax,vmin=vmin,vmax=vmax,colorbar=colorbar)
# vmin = -1
# vmax = -vmin
av_power_penduleP.plot(picks=pick,axes=axs[1,0],fmax=fmax,vmin=vmin,vmax=vmax,colorbar=colorbar)
av_power_mainP.plot(picks=pick,axes=axs[1,1],fmax=fmax,vmin=vmin,vmax=vmax,colorbar=colorbar)

av_power_mainIllusion.plot(picks=pick,axes=axs[0,2],fmax=fmax,vmin=vmin,vmax=vmax)
av_power_mainIllusionP.plot(picks=pick,axes=axs[1,2],fmax=fmax,vmin=vmin,vmax=vmax)
plt.tight_layout()


#♥all topomaps whole trial
tmin = 2.5
tmax = 25.5
fmax = 7
vmin = -0.37#-0.2
vmax = -vmin
my_cmap = discrete_cmap(13, 'RdBu_r')
fig, axs = plt.subplots(2, 3)
colorbar = False
av_power_pendule.plot_topomap(axes=axs[0,0],fmax=fmax,vlim=(vmin,vmax),tmin=tmin,tmax=tmax,cmap=my_cmap,colorbar=colorbar, cbar_fmt='%0.1f')
av_power_main.plot_topomap(axes=axs[0,1],fmax=fmax,vlim=(vmin,vmax),tmin=tmin,tmax=tmax,cmap=my_cmap,colorbar=colorbar, cbar_fmt='%0.1f')
# vmin = -0.5
# vmax = -vmin
av_power_penduleP.plot_topomap(axes=axs[1,0],fmax=fmax,vlim=(vmin,vmax),tmin=tmin,tmax=tmax,cmap=my_cmap,colorbar=colorbar, cbar_fmt='%0.1f')
av_power_mainP.plot_topomap(axes=axs[1,1],fmax=fmax,vlim=(vmin,vmax),tmin=tmin,tmax=tmax,cmap=my_cmap,colorbar=colorbar, cbar_fmt='%0.1f')

av_power_mainIllusion.plot_topomap(axes=axs[0,2],fmax=fmax,vlim=(vmin,vmax),tmin=tmin,tmax=tmax,cmap=my_cmap, cbar_fmt='%0.1f')
av_power_mainIllusionP.plot_topomap(axes=axs[1,2],fmax=fmax,vlim=(vmin,vmax),tmin=tmin,tmax=tmax,cmap=my_cmap, cbar_fmt='%0.1f')
plt.tight_layout()


#only during the vibration segments
fmax = 7
colorbar = False
t_Startvib_nf = [6.2,12.4,18.6,24.8]
durVib = 2
durTheta = 0.8
vmin = -0.15
vmax = -vmin
fig, axs = plt.subplots(2, 4)
for i in range(4):
    tmin = t_Startvib_nf[i]
    if i ==3:
        colorbar = True
    av_power_mainIllusion.plot_topomap(fmax=fmax,vlim=(vmin,vmax),axes=axs[0,i],tmin = tmin,tmax=tmin+durVib,cmap=my_cmap,colorbar=colorbar, cbar_fmt='%0.1f')
    av_power_mainIllusion.plot_topomap(fmax=fmax,vlim=(vmin,vmax),axes=axs[1,i],tmin = tmin,tmax=tmin+durTheta,cmap=my_cmap,colorbar=colorbar, cbar_fmt='%0.1f')
    
    
fmax = 7
colorbar = False
t_Startvib_nf = [6.2,12.4,18.6,24.8]
durVib = 2 #•1.3 sur la derniere 
durTheta = 0.8
fig, axs = plt.subplots(2, 4)
vmin = -0.34
vmax = -vmin
for i in range(4):
    tmin = t_Startvib_nf[i]
    if i ==3:
        colorbar = True
    av_power_mainIllusionP.plot_topomap(fmax=fmax,vlim=(vmin,vmax),axes=axs[0,i],tmin = tmin,tmax=tmin+durVib,cmap=my_cmap,colorbar=colorbar, cbar_fmt='%0.1f')
    av_power_mainIllusionP.plot_topomap(fmax=fmax,vlim=(vmin,vmax),axes=axs[1,i],tmin = tmin,tmax=tmin+durTheta,cmap=my_cmap,colorbar=colorbar, cbar_fmt='%0.1f')
    
    
    
#s'inspirer du code fig5 mais au lieu d'exclure les vib on veut exclure ailleurs
    #========== ON VIRE WHOLE TRIAL
fmax = 7
colorbar = False
t_Startvib_nf = [6.2,12.4,18.6,24.8]
durTheta = 0.8
fig, axs = plt.subplots(2, 4)
vminFB = -0.3
vmaxFB = -vminFB
for i in range(4):
    tmin = t_Startvib_nf[i]
    if i ==3:
        colorbar = True
    av_power_mainIllusion.plot_topomap(fmax=fmax,vlim=(vmin,vmax),axes=axs[0,i],tmin = tmin,tmax=tmin+durTheta,cmap=my_cmap,colorbar=colorbar, cbar_fmt='%0.1f')
    av_power_mainIllusionP.plot_topomap(fmax=fmax,vlim=(vminFB,vmaxFB),axes=axs[1,i],tmin = tmin,tmax=tmin+durTheta,cmap=my_cmap,colorbar=colorbar, cbar_fmt='%0.1f')
    