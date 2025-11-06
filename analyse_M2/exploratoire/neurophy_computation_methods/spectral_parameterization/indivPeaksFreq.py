# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:26:30 2022

@author: claire.dussard
"""
import os 
import seaborn as sns
import pathlib
import mne
import pandas as pd
#necessite d'avoir execute handleData_subject.py, et load_savedData avant 
import numpy as np 
# importer les fonctions definies par moi 
from handleData_subject import createSujetsData
from functions.load_savedData import *

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()


# os.chdir("../../../../")
# lustre_data_dir = "_RAW_DATA"
# lustre_path = pathlib.Path(lustre_data_dir)
# os.chdir(lustre_path)

# listeNumSujetsFinale.pop(1)
# listeNumSujetsFinale.pop(3)

# listeSujetsPendule_C3 = []
# listeSujetsMain_C3 = []
# listeSujetsMainIllusion_C3 = []

# for sujet in listeNumSujetsFinale:#C3 8-30Hz
    
#     mat_p = scipy.io.loadmat('../MATLAB_DATA/'+sujet+'/'+sujet+'-penduletimePooled.mat')["data"][11][5:27]
#     listeSujetsPendule_C3.append(mat_p)
#     mat_m = scipy.io.loadmat('../MATLAB_DATA/'+sujet+'/'+sujet+'-maintimePooled.mat')["data"][11][5:27]
#     listeSujetsMain_C3.append(mat_m)
#     mat_mi = scipy.io.loadmat('../MATLAB_DATA/'+sujet+'/'+sujet+'-mainIllusiontimePooled.mat')["data"][11][5:27]
#     listeSujetsMainIllusion_C3.append(mat_mi)
    
    
# test = range(3,85)
# freqRange = test[5:27] #8-30Hz

# #go through the subjects and find the freq where they desynchronized most
# arr = listeSujetsMainIllusion_C3[0]
# min_sujet_cond = min(arr)
# min_index = np.where(arr == np.amin(arr))[0][0]
# freqValue = freqRange[min_index]
# print("min freq : "+str(freqValue))
# print("min value : "+str(min_sujet_cond))
# print("mean desync"+str(arr.mean()))
# df_col=["b","g","r","c","m","y",
#         "#bc13fe","k","#5ca904","#ffcfdc","#fe01b1",
#         "#fcb001","#ff5b00","#886806","#5d1451","#41fdfe",
#         "#fedf08","#fdb147","#9d0759","#d6fffa","#feb209",
#         "#06b1c4","#f1da7a","#c3909b"]

# def findMeanFreq(listeCond):
#     listeFreqMin = []
#     listeDesyncMin = []
#     for i in range(len(listeCond)):
#         arr = listeCond[i]
#         min_sujet_cond = min(arr)
#         min_index = np.where(arr == np.amin(arr))[0][0]
#         freqValue = freqRange[min_index]
#         print("min freq : "+str(freqValue))
#         print("min value : "+str(min_sujet_cond))
#         print("mean desync"+str(arr.mean()))
#         plt.plot(freqRange,arr,label=listeNumSujetsFinale[i],color=df_col[i])
#         plt.axvline(x=freqValue,color=df_col[i])
#         listeFreqMin.append(freqValue)
#         listeDesyncMin.append(min_sujet_cond)
#     plt.legend(loc="upper right")
#     raw_signal.plot(block=True)
#     return listeFreqMin,listeDesyncMin
    
# listeFreqMin_mi,listeDesyncMin_mi = findMeanFreq(listeSujetsMainIllusion_C3)
# listeFreqMin_m,listeDesyncMin_m = findMeanFreq(listeSujetsMain_C3)
# listeFreqMin_p,listeDesyncMin_p = findMeanFreq(listeSujetsPendule_C3)
    

# print("mean minFreq desync mainVibrations"+str(np.mean(listeFreqMin_mi)))
# print("mean minFreq desync main"+str(np.mean(listeFreqMin_m)))
# print("mean minFreq desync pendule"+str(np.mean(listeFreqMin_p)))

# scipy.stats.ttest_rel(listeFreqMin_m,listeFreqMin_p)#pas de diff significative
# scipy.stats.ttest_rel(listeFreqMin_mi,listeFreqMin_m)

# print("mean mindesync mainVibrations"+str(np.mean(listeDesyncMin_mi)))
# print("mean mindesync desync main"+str(np.mean(listeDesyncMin_m)))
# print("mean mindesync desync pendule"+str(np.mean(listeDesyncMin_p)))

# scipy.stats.ttest_rel(listeDesyncMin_m,listeDesyncMin_p)#pas de diff significative
# scipy.stats.ttest_rel(listeDesyncMin_mi,listeDesyncMin_m)

# resFinal = np.zeros(shape=(23,6))
# for suj in range(23):
#     resFinal[suj][0]= listeFreqMin_p[suj]
#     resFinal[suj][1]= listeFreqMin_m[suj]
#     resFinal[suj][2]= listeFreqMin_mi[suj]
#     resFinal[suj][3]= listeDesyncMin_p[suj]
#     resFinal[suj][4]= listeDesyncMin_m[suj]
#     resFinal[suj][5]= listeDesyncMin_mi[suj]
    
# df = pd.DataFrame(resFinal,columns=["freqMin_p","freqMin_m","freqMin_mi","desyncMin_p","desyncMin_m","desyncMin_mi"])

# df.to_csv("freqIndiv_desync_condNFB.csv")

# #try to plot 3d data with a miserable fail
# av_power_main_C3 = av_power_main.data[11]
# fig = plt.figure()
# ax = fig.gca(projection='3d')

# X = yo2
# Y = yo
# X,Y = np.meshgrid(X,Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)

# surf = ax.plot_surface(X,Y,Z,
#                        linewidth=0, antialiased=False)
# raw_signal.plot(block=True)

# av_power_main.plot(picks=["C3","C4"],fmax=40,vmin=-0.26,vmax=0.26)
# raw_signal.plot(block=True)

# av_power_pendule.plot(picks=["C3","C4"],fmax=40,vmin=-0.26,vmax=0.26)
# raw_signal.plot(block=True)

# av_power_mainIllusion.plot(picks=["C3","C4"],fmax=40,vmin=-0.26,vmax=0.26)
# raw_signal.plot(block=True)