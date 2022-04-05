#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 10:51:33 2022

@author: claire.dussard
"""


liste_tfr = load_tfr_data(rawPath_main_sujets[0:2],"")

import os.path as op
 
import mne

# df = yo.to_data_frame()

# sio.savemat('test.mat',df)

data_num = yo.data

sio.savemat('test2.mat', {'data': data_num})



save_tfr_data_to_mat(load_tfr_data(rawPath_main_sujets,""),rawPath_main_sujets,"main",True,(-3,-1))

save_tfr_data_to_mat(load_tfr_data(rawPath_mainIllusion_sujets,""),rawPath_mainIllusion_sujets,"mainIllusion",True,(-3,-1))

save_tfr_data_to_mat(load_tfr_data(rawPath_pendule_sujets,""),rawPath_pendule_sujets,"pendule",True,(-3,-1))