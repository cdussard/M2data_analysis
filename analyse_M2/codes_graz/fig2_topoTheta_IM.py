# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 19:07:28 2024

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

av_power_MIalone =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/MIalone-tfr.h5")[0]

picks = "C3"
fmax = 40
vmin = -0.5
vmax = -vmin
av_power_MIalone.plot(picks=picks,fmax=fmax,vmin=vmin,vmax=vmax)

fmax = 7
vmin = -0.5
vmax = -vmin
av_power_mainIllusionP.plot_topomap(fmax=fmax,vlim=(vmin,vmax),tmin=tmin,tmax=tmax,cmap=my_cmap, cbar_fmt='%0.1f')
plt.tight_layout()