# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 18:41:50 2023

@author: claire.dussard
"""


df_global = df_global.astype(float)
df_global_pval = df_global_pval.astype(float)

#subset from fmin to fmax
fmin = 3
fmax = 40
len_to_keep = fmax - fmin
df_subset = df_global[:,0:len_to_keep+1]
df_pval_subset = df_global_pval[:,0:len_to_keep+1]
print(df_pval_subset.min())
print(df_global_pval.min())

ind_min_global=np.where(df_global_pval==df_global_pval.min())
print(ind_min_global)
ind_min_subset=np.where(df_pval_subset==df_pval_subset.min())
print(ind_min_subset)


#FDR correction

corrected_pval = mne.stats.fdr_correction(df_pval_subset)[1]
print(corrected_pval.min())
ind_min_subset_corr=np.where(corrected_pval==corrected_pval.min())
print(ind_min_subset_corr)

fmin = 3
fmax = 83
len_to_keep = fmax - fmin
df_subset = df_global[:,0:len_to_keep+1]
df_pval_subset = df_global_pval[:,0:len_to_keep+1]
print(df_pval_subset.min())
print(df_global_pval.min())
corrected_pval = mne.stats.fdr_correction(df_pval_subset)[1]

print(corrected_pval.min())
ind_min_subset_corr=np.where(corrected_pval==corrected_pval.min())
print(ind_min_subset_corr)


val_max = 
val_min =