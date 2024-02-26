#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 10:51:33 2022

@author: claire.dussard
"""


liste_tfr = load_tfr_data_windows(liste_rawPathMain,"",True)
for i in range(len(liste_tfr)):
    print(liste_tfr[i].info["ch_names"][11]=="C3")
save_elec_fr_data_to_mat(liste_tfr,rawPath_main_sujets,"main",True,(-3,-1),True)

ch_names = liste_tfr[0].info["ch_names"]
import os.path as op
import mne

# df = yo.to_data_frame()

# sio.savemat('test.mat',df)

data_num = yo.data
sio.savemat('test2.mat', {'data': data_num})
['Fp1', 'Fp2', 'F7',
 'F3', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7',
 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6',
 'P7', 'P3', 'Pz', 'P4',
 'P8', 'O1', 'Oz', 'O2', 'Fz']


# save_tfr_data_to_mat(load_tfr_data_windows(rawPath_main_sujets,"",True),rawPath_main_sujets,"main",True,(-3,-1))

# save_tfr_data_to_mat(load_tfr_data(rawPath_mainIllusion_sujets,"",True),rawPath_mainIllusion_sujets,"mainIllusion",True,(-3,-1))

# save_tfr_data_to_mat(load_tfr_data(rawPath_pendule_sujets,"",True),rawPath_pendule_sujets,"pendule",True,(-3,-1))



save_elec_fr_data_to_mat(load_tfr_data_windows(rawPath_main_sujets,"",True),rawPath_main_sujets,"main",True,(-3,-1),2.5,26.8,True)

save_elec_fr_data_to_mat(load_tfr_data_windows(rawPath_mainIllusion_sujets,"",True),rawPath_mainIllusion_sujets,"mainIllusion",True,(-3,-1),2.5,26.8,True)

save_elec_fr_data_to_mat(load_tfr_data_windows(rawPath_pendule_sujets,"",True),rawPath_pendule_sujets,"pendule",True,(-3,-1),2.5,26.8,True)

import scipy.io
mat = scipy.io.loadmat('../MATLAB_DATA/sub-S012/sub-S012-penduletimePooled.mat')
print(mat["data"].shape)