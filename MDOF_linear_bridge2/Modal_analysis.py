"""
This script is to preprocess the data to fit into the DNN model.

Written by Taeyong Kim at Ajou University.
taeyongkim@ajou.ac.kr
"""

import numpy as np
from i_MDOF_modal import MDOF_3span_modal_accel, MDOF_4span_modal_accel, MDOF_5span_modal_accel, MDOF_6span_modal_accel
from i_MDOF_modal import MDOF_3span_modal_all, MDOF_4span_modal_all, MDOF_5span_modal_all, MDOF_6span_modal_all

#%% Calculate the residual vector element, see Eq. (17) of the reference
def cal_value(Gamma_all, eigen_vect, s):
    
    mode_target = 4
    
    # If I use up to 4 eigen modes
    abc = np.zeros([len(Gamma_all),])
    for ii in range(mode_target, len(Gamma_all)):
        abc = abc+Gamma_all[ii]*eigen_vect[:,ii]

    bcd = np.zeros([len(Gamma_all),])
    for ii in range(mode_target):
        bcd = bcd+Gamma_all[ii]*eigen_vect[:,ii]
    
    abc2 = s.reshape(len(s),)-bcd   

    
    if any(abc - abc2>0.1*10**-6):
        print('Wrong error')
     
    return abc

#%% Load structural info.
p3_value = np.load('./generated_MDOF_systems/3_span.npy', allow_pickle=True)
p4_value = np.load('./generated_MDOF_systems/4_span.npy', allow_pickle=True)
p5_value = np.load('./generated_MDOF_systems/5_span.npy', allow_pickle=True)
p6_value = np.load('./generated_MDOF_systems/6_span.npy', allow_pickle=True)

#%% Modal values --> later used as inputs for the DNN model
Total_3span = []
Total_4span = []
Total_5span = []
Total_6span = []
for ii in range(2): # The number of structures considered
    tmp2_3span = p3_value[ii,:]
    tmp2_4span = p4_value[ii,:]
    tmp2_5span = p5_value[ii,:]
    tmp2_6span = p6_value[ii,:]
    
        
    tmp_results_3span = MDOF_3span_modal_all(tmp2_3span)
    tmp_results_4span = MDOF_4span_modal_all(tmp2_4span)
    tmp_results_5span = MDOF_5span_modal_all(tmp2_5span)
    tmp_results_6span = MDOF_6span_modal_all(tmp2_6span)
    
    Total_3span.append(tmp_results_3span)
    Total_4span.append(tmp_results_4span)
    Total_5span.append(tmp_results_5span)
    Total_6span.append(tmp_results_6span)
    
np.save('./generated_MDOF_systems/Modal_3span_all.npy', Total_3span) 
np.save('./generated_MDOF_systems/Modal_4span_all.npy', Total_4span) 
np.save('./generated_MDOF_systems/Modal_5span_all.npy', Total_5span) 
np.save('./generated_MDOF_systems/Modal_6span_all.npy', Total_6span) 


#%% Estimate key modal values for predicting peak acceleration, see Eq. (17) of the reference

# A total of 1485 ground motions
Total_3span = []
Total_4span = []
Total_5span = []
Total_6span = []
for ii in range(2): # The number of structures considered
    tmp2_3span = p3_value[ii,:]
    tmp2_4span = p4_value[ii,:]
    tmp2_5span = p5_value[ii,:]
    tmp2_6span = p6_value[ii,:]
    
    # Gamma_all, eigen_vect, s 
    tmp_results_3span = MDOF_3span_modal_accel(tmp2_3span)
    tmp_results_4span = MDOF_4span_modal_accel(tmp2_4span)
    tmp_results_5span = MDOF_5span_modal_accel(tmp2_5span)
    tmp_results_6span = MDOF_6span_modal_accel(tmp2_6span)
    
    # Calculate values that required
    val_3span = cal_value(tmp_results_3span[0], tmp_results_3span[1], tmp_results_3span[2])
    val_4span = cal_value(tmp_results_4span[0], tmp_results_4span[1], tmp_results_4span[2])
    val_5span = cal_value(tmp_results_5span[0], tmp_results_5span[1], tmp_results_5span[2])
    val_6span = cal_value(tmp_results_6span[0], tmp_results_6span[1], tmp_results_6span[2])
    
    Total_3span.append(np.r_[val_3span[2], val_3span[6], val_3span[10]])
    Total_4span.append(np.r_[val_4span[2], val_4span[6], val_4span[10], val_4span[14]])
    Total_5span.append(np.r_[val_5span[2], val_5span[6], val_5span[10], val_5span[14], val_5span[18]])
    Total_6span.append(np.r_[val_6span[2], val_6span[6], val_6span[10], val_6span[14], val_6span[18], val_6span[22]])
    
       
np.save('./generated_MDOF_systems/Modal_3span_accel.npy', Total_3span) 
np.save('./generated_MDOF_systems/Modal_4span_accel.npy', Total_4span) 
np.save('./generated_MDOF_systems/Modal_5span_accel.npy', Total_5span) 
np.save('./generated_MDOF_systems/Modal_6span_accel.npy', Total_6span) 

