# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:19:54 2023

@author: claire.dussard
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat

from handle_data_subject import * 

essaisMainSeule,essaisMainIllusion,essaisPendule,listeNumSujetsFinale,allSujetsDispo,listeDatesFinale,SujetsPbNomFichiers,dates,seuils_sujets = createSujetsData()
listeNumSujetsFinale.pop(1)
listeNumSujetsFinale.pop(3)

path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/rdom_scriptsData/allElecFreq_VSZero/refait_25/"


mIll = pd.read_csv(path+"dcohen_mIll.csv",header=None,delimiter=";")
main = pd.read_csv(path+"dcohen_main.csv",header=None,delimiter=";")
pend = pd.read_csv(path+"dcohen_pendule.csv",header=None,delimiter=";")

path = "../../../../MATLAB_DATA/"

all_data = ["pendule","main","mainIllusion"]

channels=["Fp1","Fp2","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6","T7","C3","Cz","C4","T8",
"CP5","CP1","CP2","CP6","P7","P3","Pz","P4","P8","O1","Oz","O2"]
freqs = np.arange(3,84,1)

n_elec = len(channels)
n_freq = len(freqs)

i = 0
dict_total = pd.DataFrame(columns = ["num_sujet","FB","elec","freq","ERD_value"])
for suj in listeNumSujetsFinale:
    for FB in all_data:
        path_sujet_fb = path+suj+"/"+suj+"-"+FB+"timePooled.mat"
        print(path_sujet_fb)
        data = loadmat(path_sujet_fb)["data"]
        for elec_i in range(n_elec):
           for freq_i in range(n_freq):
               value = data[elec_i,freq_i]
               dict_suj = {"num_sujet": allSujetsDispo[i],
                "FB": FB,
                "elec":channels[elec_i],
                "freq":freq_i+3,
                "ERD_value":value }
               print(dict_suj)
               dict_total = dict_total.append(dict_suj,ignore_index=True)

    i += 1

dict_total.to_csv("dictCSV_agencyCorrel.csv")
    
#versionopti ?=================

import numpy as np
import pandas as pd

all_data = ["pendule", "main", "mainIllusion"]
channels = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
            "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]
freqs = np.arange(3, 84, 1)

n_elec = len(channels)
n_freq = len(freqs)

num_iterations = len(listeNumSujetsFinale) * len(all_data) * n_elec * n_freq
dict_total = np.empty((num_iterations, 5), dtype=object)
index = 0

for suj in listeNumSujetsFinale:
    for FB in all_data:
        path_sujet_fb = path + suj + "/" + suj + "-" + FB + "timePooled.mat"
        print(path_sujet_fb)
        data = loadmat(path_sujet_fb)["data"]

        for elec_i in range(n_elec):
            for freq_i in range(n_freq):
                value = data[elec_i, freq_i]
                dict_total[index] = [suj, FB, channels[elec_i], freq_i + 3, value]
                index += 1

dict_total = pd.DataFrame(dict_total[:index], columns=["num_sujet", "FB", "elec", "freq", "ERD_value"])
         
             
dict_total.to_csv("optimize_dataCorrel.csv")      


#PLOT


import pandas as pd
import numpy as np
import imagesc

# Path to the directory containing the CSV files
path = "C:/Users/claire.dussard/OneDrive - ICM/Bureau/fig_brian/correlERD_agency_colorgrid/"

# Read the CSV files
mIll = pd.read_csv(path + "df_main_pval.csv", header=None, delimiter=",").iloc[:, 1:].values
main = pd.read_csv(path + "df_mainvib_pval.csv", header=None, delimiter=",").iloc[:, 1:].values
pend = pd.read_csv(path + "df_pendule_pval.csv", header=None, delimiter=",").iloc[:, 1:].values

# Convert to numeric values and float dtype
mIll = pd.to_numeric(mIll.flatten(), errors='coerce').reshape(mIll.shape).astype(float)
main = pd.to_numeric(main.flatten(), errors='coerce').reshape(main.shape).astype(float)
pend = pd.to_numeric(pend.flatten(), errors='coerce').reshape(pend.shape).astype(float)

# Plot the color grid maps
imagesc.plot(pend, cmap="Blues")
imagesc.plot(main, cmap="Blues")
imagesc.plot(mIll, cmap="Blues")




        