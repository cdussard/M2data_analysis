# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 12:39:59 2023

@author: claire.dussard
"""
import pandas as pd
data_impedances = pd.DataFrame(columns=["num_sujet","elec","impedance"])

for i in range(23):
    num_sujet = i
    liste_sujets_recordings = read_raw_data(liste_rawPath_rawRest[num_sujet:num_sujet+1])
    ch_names =  ['Ref','Gnd','Fp1', 'Fp2', 'F7', 'F3','F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10','T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5','CP1','CP2','CP6','TP10','P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
    
    imp_sujet = liste_sujets_recordings[0].impedances
    for ch in ch_names:
        try:
            impedance = imp_sujet[ch]["imp"]
        except:
            print("not found")
            impedance = None
            
        data_ch = {"num_sujet":num_sujet,
                   "elec":ch,
                   "impedance":impedance
            
            }
        data_impedances = data_impedances.append(data_ch,ignore_index=True)
        
    print(data_impedances)
    
data_impedances.to_csv("../csv_files/data_impedances.csv")