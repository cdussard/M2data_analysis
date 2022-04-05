#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:54:20 2021

@author: claire.dussard
"""

channelsSansFz = ['Fp1', 'Fp2', 'F7', 'F3','F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10','T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5','CP1','CP2','CP6','TP10','P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2','HEOG','VEOG']

channelsAvecFz = ['Fp1', 'Fp2', 'F7', 'F3','Fz','F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10','T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5','CP1','CP2','CP6','TP10','P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2','HEOG','VEOG']

for signal in listeFilteredICA[0:2]:
    bonOrdreEEG=signal.copy().reorder_channels(channelsSansFz)
    bonOrdreEEG.plot(block=True)