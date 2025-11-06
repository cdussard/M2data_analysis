# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:39:39 2022

@author: claire.dussard
"""

from functions.plotPsd import *


#load data
gd_average_exec_noBL =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/exec_noBaseline-tfr.h5")[0]
gd_average_exec =  mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/exec-tfr.h5")[0]
av_power_pendule = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule-tfr.h5")[0]
av_power_main = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main-tfr.h5")[0] 
av_power_pendule_noBL = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/pendule_noBaseline-tfr.h5")[0]
av_power_main_noBL = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/main_noBaseline-tfr.h5")[0] 
av_power_mainIllusion_noBL = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion_noBaseline-tfr.h5")[0] 
av_power_mainIllusion = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/mainIllusion-tfr.h5")[0] 
av_power_Rest_noBL = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/rest_noBaseline-tfr.h5")[0]
av_power_Rest = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/rest-tfr.h5")[0]
Rest_moinsExec  = av_power_Rest_noBL - gd_average_exec_noBL 
Rest_moinsNFB = av_power_Rest_noBL - av_power_main_noBL 

#data control passif
rest_main = mne.time_frequency.read_tfrs("../AV_TFR/all_sujets/effet_main-tfr.h5")[0]
plot_elec_cond(rest_main,"C3","rest_main",11,freqs,fig,ax,scaleMin,scaleMax,10,25,fmin,fmax,'green',True)


#plot data
fig, ax = plt.subplots()
freqs = np.arange(3, 85, 1)
# scaleMin = 0
# scaleMax = 1e-9
scaleMin = -0.4
scaleMax = 0.3
fmin = 3 
fmax = 70
plot_elec_cond(av_power_pendule,"C3","NFBpendule",11,freqs,fig,ax,scaleMin,scaleMax,10,25,fmin,fmax,'green',True)
plot_elec_cond(gd_average_exec,"C3","exec",11,freqs,fig,ax,scaleMin,scaleMax,10,25,fmin,fmax)
plot_elec_cond(av_power_Rest,"C3","rest",11,freqs,fig,ax,scaleMin,scaleMax,10,25,fmin,fmax)
plot_elec_cond(av_power_main,"C3","NFBmain",11,freqs,fig,ax,scaleMin,scaleMax,10,25,fmin,fmax)
raw_signal.plot(block=True)

#no BL
fig, ax = plt.subplots()
freqs = np.arange(3, 85, 1)
scaleMin = -1e-10
scaleMax = 1e-10
fmin = 3 
fmax = 40
plot_elec_cond(Rest_moinsExec,"C3","rest-exec",11,freqs,fig,ax,scaleMin,scaleMax,10,25,fmin,fmax)
plot_elec_cond(gd_average_exec_noBL,"C3","exec",11,freqs,fig,ax,scaleMin,scaleMax,10,25,fmin,fmax)
plot_elec_cond(av_power_Rest_noBL,"C3","rest",11,freqs,fig,ax,scaleMin,scaleMax,10,25,fmin,fmax)
plot_elec_cond(av_power_main_noBL,"C3","NFBmain",11,freqs,fig,ax,scaleMin,scaleMax,10,25,fmin,fmax)
#plot_elec_cond(av_power_pendule_noBL,"C3","C3NFBpendule",11,freqs,fig,ax,scaleMin,scaleMax,5,25,fmin,fmax)
plot_elec_cond(Rest_moinsNFB,"C3","rest-NFBmain",11,freqs,fig,ax,scaleMin,scaleMax,10,25,fmin,fmax)
#plot_elec_cond(gd_average_exec_noBL,"C4","C4exec",13,freqs,fig,ax,scaleMin,scaleMax,10,25,fmin,fmax)
#plot_elec_cond(av_power_Rest_noBL,"C4","C4rest",13,freqs,fig,ax,scaleMin,scaleMax,10,25,fmin,fmax)
#plot_elec_cond(av_power_pendule_noBL,"C4","C4NFBpendule",13,freqs,fig,ax,scaleMin,scaleMax,5,25,fmin,fmax)
#plot_elec_cond(av_power_main_noBL,"C4","C4NFBmain",13,freqs,fig,ax,scaleMin,scaleMax,10,25,fmin,fmax)
raw_signal.plot(block=True)


fig, ax = plt.subplots()
i = 0
scaleMin = 0.01e-9
scaleMax = 0.85e-9
fmin = 5 
fmax = 35
for tfr in liste_power_sujets[1:]:
    plot_elec_cond(tfr,"C3","rest"+str(allSujetsDispo[2:][i]),11,freqs,fig,ax&,scaleMin,scaleMax,10,25,fmin,fmax,'lightgrey',False)
    i += 1
plot_elec_cond(av_power_Rest,"C3","restmoy",11,freqs,fig,ax,scaleMin,scaleMax,10,25,fmin,fmax,'black',True)
plot_elec_cond(av_power_main_noBL,"C3","NFBmoy_main",11,freqs,fig,ax,scaleMin,scaleMax,10,25,fmin,fmax,'blue',True)
plot_elec_cond(gd_average_exec_noBL,"C3","execmoy",11,freqs,fig,ax,scaleMin,scaleMax,10,25,fmin,fmax,'green',True)
#plot_elec_cond(av_power_pendule_noBL,"C3","NFBmoy_pendule",11,freqs,fig,ax,scaleMin,scaleMax,10,25,fmin,fmax,'green',True)
#plot_elec_cond(av_power_mainIllusion_noBL,"C3","NFBmoy_mainIllusion",11,freqs,fig,ax,scaleMin,scaleMax,10,25,fmin,fmax,'red',True)
 
   
raw_signal.plot(block=True)

#indiv
fig, ax = plt.subplots()
scaleMin = 0
scaleMax = 1e-9
fmin = 5 
fmax = 35
i = 22
for tfr in liste_power_sujets[i:i+5]:
    plot_elec_cond(tfr,"C3","rest"+str(allSujetsDispo[2:][i]),11,freqs,fig,ax,scaleMin,scaleMax,10,25,fmin,fmax,None,True)
    i += 1
raw_signal.plot(block=True)