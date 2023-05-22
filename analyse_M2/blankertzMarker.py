# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 16:32:54 2022

@author: claire.dussard
"""
#compute blankertz marker# libraries
import numpy as np
import pandas as pd
import os, mne, ast,fnmatch,scipy
import warnings
from scipy.signal import find_peaks
from scipy import stats
import matplotlib.pyplot as plt
#from pyprep.noisy import Noisydata

from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs

# import baseline files:
fileslist_oe = [] #list of path to gdf baseline files (open eyes)
filesnames_oe = [] #list of baseline files names (open eyes)
subjects_id = [] #list of IDs 
corrected_raw = pd.DataFrame()
kw_oe = '*oo*' 
directory_path = r'//l2export/iss02.cenir/analyse/meeg/BETAPARK/_RAW_DATA'#modif
laplacian_channels = ['C3','C4']
laplacians_left = ['CP5','CP1','FC5', 'FC1']
laplacians_right = ['FC2', 'FC6','CP2','CP6']

from numpy import ones, array
import scipy
from numpy.linalg import norm, lstsq
import scipy.stats as stats
import numpy as np
import warnings


# MYFITFUN(lambda,t,y) returns the error between the data and the values
# computed by the current function of lambda.
#
# MYFITFUN assumes a function of the form
#
#   y =  + c(1) + c(2) / t^lambda
#
# with linear parameters c(i) and nonlinear parameter lambda.


def minimize_fuc1(lambd, f, y):
    a = ones([len(f), 2])
    a[:, 1] = 1. / (f ** lambd)

    c = lstsq(a, y, rcond=None)[0]
    z = a.dot(c)
    err = norm(z - y)
    return err


def minimize_fuc2(params, f, y):
    mu = array([0., 0.])
    sigma = array([0., 0.])
    lambd = params[0]
    mu[0] = params[1]
    mu[1] = params[2]
    sigma[0] = params[3]
    sigma[1] = params[4]

    a = ones([len(f), 4])
    a[:, 1] = 1. / (f ** lambd)
    a[:, 2] = stats.norm.pdf(f, mu[0], sigma[0])
    a[:, 3] = stats.norm.pdf(f, mu[1], sigma[1])

    c = scipy.linalg.lstsq(a, y)[0]
    z = a.dot(c)
    err = norm(z - y)
    return err

def minimize_fuc3(params, f, y):
    mu = array([0., 0.])
    sigma = array([0., 0.])
    lambd = params[0]
    mu[0] = params[1]
    mu[1] = params[2]
    sigma[0] = params[3]
    sigma[1] = params[4]

    a = ones([len(f), 4])
    a[:, 1] = 1. / (f ** lambd)
    a[:, 2] = stats.norm.pdf(f, mu[0], sigma[0])
    a[:, 3] = stats.norm.pdf(f, mu[1], sigma[1])
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            col_mean = np.nanmean(a, axis=0)
        except RuntimeWarning:
            col_mean = [0.3, 10.0, 20.0, 1.0, 3.0]  

    #Find indicies that you need to replace
    inds = np.where(np.isnan(a))
    #Place column means in the indices. Align the arrays using take
    a[inds] = np.take(col_mean, inds[1])

    c = scipy.linalg.lstsq(a, y)[0]
    z = a.dot(c)
    err = norm(z - y)
    return err

def moving_av(signal, window, method):
    m = np.size(signal)
    signal = signal.reshape(m)
    if 'quick' in method:
        new_signal = np.zeros_like(signal)
        new_signal[0] = signal[0]
        for i in range(1, window):
            new_signal[i] = (new_signal[i - 1] * i + signal[i]) / (i + 1)
        for i in range(window, m):
            new_signal[i] = new_signal[i - 1] + (signal[i] - signal[i - window]) / window
    elif 'center' in method:
        new_signal = np.zeros_like(signal)
        w0 = int(-np.ceil((window - 1) / 2))
        w1 = int(window - 1 + w0)
        new_signal[0] = np.mean(signal[0: w1 + 1])
        for i in range(1, -w0 + 1):
            new_signal[i] = (new_signal[i - 1] * (w1 + i + 1 - 1) + signal[i + w1]) / (w1 + i + 1)
        for j in range(-w0 + 1, m - w1):
            new_signal[j] = new_signal[j - 1] + (signal[j + w1] - signal[j + w0 - 1]) / window
        for i0 in range(1, w1 + 1):
            i = m - w1 + i0 - 1
            new_signal[i] = (new_signal[i - 1] * (window - i0 + 1) - (signal[i + w0 - 1])) / (window - i0)
    else:
        new_signal = signal
        print('no method were specify')

    return new_signal
def compute_laplacian(epoch_oe,picks_amp=abs(120e-06)):
    laplacian_dict={} # creation of the laplacian dictionary (2 laplacians for each subject)
    laplacian = {'C3':['CP5','CP1','FC5', 'FC1'],'C4':['FC2', 'FC6','CP2','CP6']}#modified
    weird_subjects = []
    for subject, epochs in epoch_oe.items(): #epoch dictionnary 
        new_laplacian_epochs={}
        centers = [c for c in laplacian.keys()]
        print("sujet : "+str(subject))
        for center, laplacians in laplacian.items():
            new_epoch = 4*epochs.copy().pick_channels([center]).get_data()
            for i in range(len(laplacians)):
                new_epoch = new_epoch-epochs.copy().pick_channels([laplacians[i]]).get_data()
                new_epoch_data = new_epoch
            # remove=[]
            # for ep in range(new_epoch.shape[0]):                       
            #     for ampli in new_epoch[ep][0]:
            #         if (ampli > picks_amp): 
            #             remove.append(ep)
            # if (len(np.unique(remove)) < new_epoch.shape[0] ):
            #     new_epoch_data = np.delete(new_epoch,np.unique(remove),axis=0)   
            # else:
            #     weird_subjects.append(subject)
            #     new_epoch_data = new_epoch 

            info = mne.create_info(ch_names=[center], sfreq=epochs.info['sfreq'], ch_types='eeg')
            print("len epochs: "+str(len(new_epoch_data)))
            new_epoch = mne.EpochsArray(data=new_epoch_data, info=info, verbose=0)

            new_laplacian_epochs[center] = new_epoch                    
        laplacian_dict[subject]= new_laplacian_epochs
#    print('WEIRD SUBJECTS: ' + str(np.unique(weird_subjects)))
    return laplacian_dict

def compute_psd(laplacian_dict):
    laplacian = {'C3':['CP5','CP1','FC5', 'FC1'],'C4':['FC2', 'FC6','CP2','CP6']}#modified
    psd_dict = {}
    for subject,epochs in laplacian_dict.items():
        temporary_dict = {}
        for center in laplacian.keys():
            epoch = laplacian_dict[subject][center]
            kwargs = dict(fmin=2, fmax=30, n_jobs=1)
            psds,freqs = mne.time_frequency.psd_multitaper(epoch, bandwidth=1,**kwargs)
            psds =10*np.log10(psds)
            temporary_dict[center]=[psds,freqs]
        psd_dict[subject]= temporary_dict
    compute_predictor_mean(psd_dict)
    print("Computation done!!!!")
    return psd_dict
    
def compute_predictor_fit_functions(psds,freqs,subject,center):        
    x0 = [0.3, 10.0, 20.0, 1.0, 3.0]
    subj_not_converge_mean = []
    try:
        estimated_params = scipy.optimize.fmin(minimize_fuc2, disp=False, maxiter=100000,
                            maxfun=10000, x0=x0,args=(freqs, psds))
    except ValueError:

        try:
            estimated_params = scipy.optimize.fmin(minimize_fuc3, disp=False, maxiter=100000,
                            maxfun=10000, x0=x0,args=(freqs, psds))

        except ValueError:
            print(str([subject,center])+': toujour pas')
            subj_not_converge_mean.append([subject,center])
            result=[]
        else:
            lambd = estimated_params[0]
            mu = estimated_params[1:3]
            sigma = estimated_params[3:]
            a = np.ones([len(freqs), 4])
            a[:, 1] = 1. / (freqs ** lambd)
            a[:, 2] = stats.norm.pdf(freqs, mu[0], sigma[0])
            a[:, 3] = stats.norm.pdf(freqs, mu[1], sigma[1])
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    col_mean = np.nanmean(a, axis=0)
                except RuntimeWarning:
                    print(str([subject,center])+': remplacement par valeur defaut')
                    col_mean = [0.3, 10.0, 20.0, 1.0, 3.0]
                    #Find indicies that you need to replace
                    inds = np.where(np.isnan(a))
                    #Place column means in the indices. Align the arrays using take
                    a[inds] = np.take(col_mean, inds[1])

                    result=find_fit_functions(a,estimated_params,psds,freqs)

                else:
                    print(str([subject,center])+': remplacement par moyenne')


                    #Find indicies that you need to replace
                    inds = np.where(np.isnan(a))
                    #Place column means in the indices. Align the arrays using take
                    a[inds] = np.take(col_mean, inds[1])

                    result=find_fit_functions(a,estimated_params,psds,freqs)

    else:
        lambd = estimated_params[0]
        mu = estimated_params[1:3]
        sigma = estimated_params[3:]
        a = np.ones([len(freqs), 4])
        a[:, 1] = 1. / (freqs ** lambd)
        a[:, 2] = stats.norm.pdf(freqs, mu[0], sigma[0])
        a[:, 3] = stats.norm.pdf(freqs, mu[1], sigma[1])#pb
        result=find_fit_functions(a,estimated_params,psds,freqs)

    return (result)

def find_fit_functions(a,estimated_params,psds,freqs):
    lambd = estimated_params[0]
    mu = estimated_params[1:3]
    sigma = estimated_params[3:]
    k = scipy.linalg.lstsq(a, psds)[0]
    #noise floor
    g1 = (k[0]+(k[1]/freqs**lambd))
    #fitted values
    g = (k[0] + (k[1] / freqs ** lambd)) + (k[2] * stats.norm.pdf(freqs, mu[0], sigma[0])) + (
                    k[3] * stats.norm.pdf(freqs, mu[1],sigma[1]))


    if len(psds) == 0:
        result = []
    else:
        first_peak_start = int(np.argwhere((psds[:-1]) < (psds[1:]))[0])


        lowest_spec = np.argmin(psds)
        if lowest_spec == first_peak_start: #fail to find a minimum
            lowest_spec = int(np.argwhere(psds == psds[-1])[0][0])
        new_freqs = [freqs[0], freqs[first_peak_start], freqs[lowest_spec]]
        new_psds = psds[[0, first_peak_start, lowest_spec]]


        estimated_lambda = scipy.optimize.fload_data_postICA_postdropBadmin(minimize_fuc1, disp=False, maxiter=10000, maxfun=100000, x0=1.,
                                            args=(new_freqs, new_psds))#ne fonctionne pas, je n'ai pas cette fonction
        a = np.ones([len(new_freqs), 2])
        a[:, 1] = 1 / (new_freqs ** estimated_lambda)
        k = lstsq(a, new_psds, rcond=None)[0]

        # calculate the floor noise function
        yhat2 = k[0] + k[1] / freqs ** estimated_lambda

        result=[g1,g,yhat2,sigma[0],sigma[1],estimated_lambda]
    return(result)   

def peak_information(freqs,psds,g,yhat2,sigma,fmin,fmax):
    mask = np.where((freqs >= fmin) & (freqs <= fmax))
    warning=0
    stuetz=np.where(yhat2[mask]<=g[mask])[0]
    stuetz += np.argwhere((freqs >= fmin))[0][0]

    if len(stuetz)<2:
        result=[0,0,0,0,0,freqs,psds,g,yhat2]
    else:         
        if g[mask][0]<g[mask][-1]:
           warning=1

        addi = np.max(g[stuetz]-yhat2[stuetz])
        posi = np.argwhere(g-yhat2 == (g[stuetz]-yhat2[stuetz]).max())[0][0]
        pick_width,pick_ampli_var = pick_features(freqs,g,yhat2,addi,posi,sigma)


        if (addi>(np.max(psds)-np.min(psds))) or (addi<0) or (np.isnan(addi)):
            result=[0,0,0,0,0,freqs,psds,g,yhat2]
        else:
            result=[posi,addi,warning,pick_width,pick_ampli_var,freqs,psds,g,yhat2]
    return result
def pick_features(freqs,g,yhat2,addi,posi,sigma):
    _,_,fmin_idx,fmax_idx=scipy.signal.peak_widths(g-yhat2, [posi], rel_height=0.5)
    #width
    pick_width = freqs[int(round(fmax_idx[0]))]-freqs[int(round(fmin_idx[0]))] #sigma

    #ampliture variance
    pick_ampli_var = np.var((g-yhat2)[int(fmin_idx[0]):int(fmax_idx[0])+1])
    #fmin = freqs[posi]-sigma/2
    #fmax = freqs[posi]+sigma/2
    #idx_fmin = np.argmin(np.abs(freqs - fmin))
    #idx_fmax = np.argmin(np.abs(freqs - fmax))
    #pick_ampli_var = np.var((g-yhat2)[idx_fmin:idx_fmax+1])


    return pick_width,pick_ampli_var

def compute_predictor_mean(psd_dict):
    fmin_alpha = int(5)
    fmax_alpha = int(15)
    fmin_beta = int(15)
    fmax_beta = int(30)
    subj_not_converge_mean=[]
    pred_dict_mean={}
    peak_info_alpha ={}
    peak_info_beta ={}
    subject_weird_psd = []
    subject_failed_predictor = []
    for subject,psd in psd_dict.items():
        mu_pred = []
        beta_pred=[]
        estimated_lambda = []
        temporary_dict={}
        temporary_dict_alpha = {}
        temporary_dict_beta = {}
        for center,psd_multitaper in psd.items():
            psds = psd_multitaper[0]
            freqs = psd_multitaper[1]            
            psds_mean=psds.mean(axis=0)[0]
            print([subject,center])
            temporary_dict[center]= compute_predictor_fit_functions(psds_mean,freqs,subject,center)
            if len(temporary_dict[center])==0:# fail in optimization raised
                subject_failed_predictor.append([subject,center])
                estimated_lambda.append(0)
                print(str([subject,center])+' fail 1')
            if len(temporary_dict[center])!=0: #si on a eu convergence
                temporary_dict_alpha[center]=peak_information(freqs,psds_mean,temporary_dict[center][1],temporary_dict[center][2],temporary_dict[center][3],fmin_alpha,fmax_alpha)
                temporary_dict_beta[center]=peak_information(freqs,psds_mean,temporary_dict[center][1],temporary_dict[center][2],temporary_dict[center][4],fmin_beta,fmax_beta)
                if (np.mean(temporary_dict_alpha[center][:5])==0) or (np.mean(temporary_dict_beta[center][:5])==0):
                    subject_failed_predictor.append([subject,center])
                    estimated_lambda.append(temporary_dict[center][5])

                    if (np.mean(temporary_dict_alpha[center][:5])==0) and (np.mean(temporary_dict_beta[center][:5])!=0):
                        #mu_pred.append(0)
                        estimated_lambda.append(temporary_dict[center][5])
                        beta_pred.append(temporary_dict_beta[center][1])

                    elif (np.mean(temporary_dict_alpha[center][:5])!=0) and (np.mean(temporary_dict_beta[center][:5])==0):
                        #beta_pred.append(0)
                        estimated_lambda.append(temporary_dict[center][5])
                        mu_pred.append(temporary_dict_alpha[center][1])

                else:
                    estimated_lambda.append(temporary_dict[center][5])
                    mu_pred.append(temporary_dict_alpha[center][1])
                    beta_pred.append(temporary_dict_beta[center][1])


                if temporary_dict_alpha[center][2]==1:
                    subject_weird_psd.append([subject,center])
                    #mu_pred.append(temporary_dict_alpha[center][1])
                    #beta_pred.append(temporary_dict_beta[center][1])


            else:
                estimated_lambda.append(0)
                temporary_dict_alpha[center]=0
                temporary_dict_beta[center]=0
        temporary_dict_alpha['mu_pred']=np.mean(mu_pred)
        temporary_dict_alpha['estimated_lambda']=np.mean(estimated_lambda)
        temporary_dict_beta['beta_pred']=np.mean(beta_pred)
        pred_dict_mean[subject]= temporary_dict # {'C4':[g1,g,yhta2,f],'C3':[g1,g,yhat2,f]}
        peak_info_alpha[subject]= temporary_dict_alpha
        peak_info_beta[subject]= temporary_dict_beta
        
        
    # dataframe creation
    df_data=[]
    #mu pred
    for subject,general_dict in peak_info_alpha.items():
        columns=['ID']
        data = [subject]
        mean_f_mu = []
        mean_mu_width = []
        mean_mu_var_ampli = []
        for center in general_dict.keys():
            if center == 'mu_pred':
                mu_pred = general_dict[center] #on recupère le mu_pred
                columns.extend(['mu_pred'])
                data.extend([mu_pred])
            elif center == 'estimated_lambda':
                estimated_lambda = general_dict[center] #on recupère le estimated lambda
                columns.extend(['estimated_lambda'])
                data.extend([estimated_lambda])
            else:
                center_mu = general_dict[center][1] #addi
                if general_dict[center][0]==0:
                    center_f_mu=0
                else:
                    center_f_mu = general_dict[center][5][general_dict[center][0]] #freqs[posi]
                center_mu_width = general_dict[center][3] #width
                center_mu_var_ampli = general_dict[center][4] #ampli var in width

                mean_f_mu.append(center_f_mu)
                mean_mu_width.append(center_mu_width)
                mean_mu_var_ampli.append(center_mu_var_ampli)

                data.extend([center_mu,center_f_mu,center_mu_width,center_mu_var_ampli])
                columns.extend([center+'-mu',center+'-f-mu',center+'-mu-width',center+'-mu-var-ampli'])
        data.extend([np.mean(mean_f_mu),np.mean(mean_mu_width),np.mean(mean_mu_var_ampli),])
        columns.extend(['mean_f_mu','mean_mu_width','mean_mu_var_ampli'])
        df_data.append(data)
    df1 = pd.DataFrame(df_data,columns=columns)
    #beta pred
    df_data=[]
    for subject,general_dict in peak_info_beta.items():
        columns=[]
        data = []
        mean_f_beta = []
        mean_beta_width = []
        mean_beta_var_ampli = []
        for center in general_dict.keys():
            if center == 'beta_pred':
                beta_pred = general_dict[center] #on recupère le mu_pred
                columns.extend(['beta_pred'])
                data.extend([beta_pred])
            else:
                center_beta = general_dict[center][1] #addi
                if general_dict[center][0]==0:
                    center_f_beta=0
                else:
                    center_f_beta = general_dict[center][5][general_dict[center][0]] #freqs[posi]
                center_beta_width = general_dict[center][3] #width
                center_beta_var_ampli = general_dict[center][4] #ampli var in width

                mean_f_beta.append(center_f_beta)
                mean_beta_width.append(center_beta_width)
                mean_beta_var_ampli.append(center_beta_var_ampli)
                data.extend([center_beta,center_f_beta,center_beta_width,center_beta_var_ampli])
                columns.extend([center+'-beta',center+'-f-beta',center+'-beta-width',center+'-beta-var-ampli'])
        data.extend([np.mean(mean_f_beta),np.mean(mean_beta_width),np.mean(mean_beta_var_ampli)])
        columns.extend(['mean_f_beta','mean_beta_width','mean_beta_var_ampli'])

        df_data.append(data)
    df2 = pd.DataFrame(df_data,columns=columns)

    result = pd.concat([df1, df2], axis=1)

    return result

def create_epoch(listeEpochs):
    picks = ['C5','C3','C1','CP3','FC3','C2','C4','C6','CP4','FC4'] # pas utilise pour instant
    # Open EYES
    epoch_oe = {}
    new_id =[]
    for iD in range(len(listeEpochs)):
        if iD ==6:
            pass
        else:
            epoch = listeEpochs[iD]
            new_id.append(iD)
            epoch_oe[iD]=epoch
    return epoch_oe

epoch_oe = create_epoch(liste_epochs_averageRef_rest)

#get epoch_oe
# def Convert(lst):
#     res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
#     return res_dct
# epoch_oe = Convert(liste_epochs_averageRef_rest) #marche pas 

laplacian_dict = compute_laplacian(epoch_oe,picks_amp=abs(120e-06))

psd_dict = compute_psd(laplacian_dict)
result_df = compute_predictor_mean(psd_dict)