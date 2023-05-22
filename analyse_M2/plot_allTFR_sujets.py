# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 12:10:31 2022

@author: claire.dussard
"""

import matplotlib.pyplot as plt 
from functions.load_savedData import *

listeTfrAv_main = load_tfr_data_windows(liste_rawPathMain,"",True)
listeTfrAv_mainIllusion = load_tfr_data_windows(liste_rawPathMainIllusion,"",True)
listeTfrAv_pendule = load_tfr_data_windows(liste_rawPathPendule,"",True)


my_cmap = discrete_cmap(13, 'Reds')
baseline = (-3,-1)
vmin = 0
vmax = 3#0.3e-9

mode = "ratio"
x = 4
y = 3
fig, axs = plt.subplots(x,y)
num_suj = 0
for i in range(x):
    for j in range(y):
        axes = axs[i,j]
        if num_suj>22:
            pass
        else:
            listeTfrAv_main[num_suj].plot(picks="C3",baseline=baseline,mode=mode,fmax=40,axes=axes,cmap = my_cmap, vmin = vmin, vmax = vmax,tmin=None)
            num_suj += 1 
        
raw_signal.plot(block=True)
        

#compare BL VS no BL
x = 4
y = 2
fig, axs = plt.subplots(x,y)
num_suj = 12
for i in range(x):
    for j in range(y):
        if (num_suj/x)==5:
            print(x)
            print("pass")
            pass
        else:
            axes = axs[i,0]
            listeTfrAv_main[num_suj].plot(picks="C3",baseline=None,fmax=40,axes=axes,cmap = my_cmap, vmin = 0, vmax = 0.3e-9,tmin=None,colorbar=False)
            axes = axs[i,1]
            listeTfrAv_main[num_suj].plot(picks="C3",baseline=(-3,-1),mode="ratio",fmax=40,axes=axes,cmap = my_cmap, vmin = 0, vmax = 3,tmin=None,colorbar=False)
            num_suj += 1 
        
raw_signal.plot(block=True)

liste_val_NFB = []
liste_val_BL = []
for i in range(len(listeTfrAv_main)):
#compute correlation between logratio value and baseline value in the 12-15Hz band
    #listeTfrAv_main[i].data[11][10:14]#12-15Hz data
    #value BL  : -3,-1
    val_BL = np.mean(listeTfrAv_main[i].data[11][10:14][:,500:1000],axis = 0)
    #value NFB : 2.5,26.8
    val_NFB = np.mean(listeTfrAv_main[i].data[11][10:14][:,1875:7950],axis=0)
    liste_val_NFB.append(np.mean(val_NFB))
    liste_val_BL.append(np.mean(val_BL))
    #plt.plot(val_BL)
    #plt.plot(val_NFB)
    #raw_signal.plot(block=True)
    
    
plt.scatter(liste_val_NFB,liste_val_BL)
raw_signal.plot(block=True)


    

