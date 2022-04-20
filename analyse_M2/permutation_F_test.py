# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 19:22:01 2022

@author: claire.dussard
"""
import pandas as pd
import mne
douzeQuinzeHzData = pd.read_csv("./data/Jasp_anova/ANOVA_12_15Hz_C3_long.csv")
df_douzeQuinzeHzData=douzeQuinzeHzData.iloc[: , 1:]
df_douzeQuinzeHzData_mP = df_douzeQuinzeHzData.iloc[:,0:2]
mne.stats.permutation_t_test(df_douzeQuinzeHzData_mP, n_permutations=10000, tail=0)
*
df_douzeQuinzeHzData_mMi = df_douzeQuinzeHzData.iloc[:,1:3]
mne.stats.permutation_t_test(df_douzeQuinzeHzData_mMi, n_permutations=10000, tail=0)