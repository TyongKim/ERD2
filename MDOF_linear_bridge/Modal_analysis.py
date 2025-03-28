"""
This script is to preprocess the data to fit into the DNN model.

Written by Taeyong Kim at Ajou University.
taeyongkim@ajou.ac.kr
"""

import numpy as np
from i_MDOF_modal import MDOF_3span_modal_accel, MDOF_4span_modal_accel, MDOF_5span_modal_accel, MDOF_6span_modal_accel
from i_MDOF_modal import MDOF_3span_modal_all, MDOF_4span_modal_all, MDOF_5span_modal_all, MDOF_6span_modal_all
import pickle

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
# p3_value = np.load('./generated_MDOF_systems/3_span.npy', allow_pickle=True)
# p4_value = np.load('./generated_MDOF_systems/4_span.npy', allow_pickle=True)
# p5_value = np.load('./generated_MDOF_systems/5_span.npy', allow_pickle=True)
# p6_value = np.load('./generated_MDOF_systems/6_span.npy', allow_pickle=True)

with open('./generated_MDOF_systems/Example_3_span.pickle', 'rb') as f:
    p3_value = pickle.load(f)
    
with open('./generated_MDOF_systems/Example_4_span.pickle', 'rb') as f:
    p4_value = pickle.load(f)
    
with open('./generated_MDOF_systems/Example_5_span.pickle', 'rb') as f:
    p5_value = pickle.load(f)
    
with open('./generated_MDOF_systems/Example_6_span.pickle', 'rb') as f:
    p6_value = pickle.load(f)

p3_value = p3_value[0][0]
p4_value = p4_value[0][0]
p5_value = p5_value[0][0]
p6_value = p6_value[0][0]

#%% Modal values --> later used as inputs for the DNN model

Total_3span = MDOF_3span_modal_all(p3_value)
Total_4span = MDOF_4span_modal_all(p4_value)
Total_5span = MDOF_5span_modal_all(p5_value)
Total_6span = MDOF_6span_modal_all(p6_value)

np.save('./generated_MDOF_systems/Example_Modal_3span_all.npy', Total_3span) 
np.save('./generated_MDOF_systems/Example_Modal_4span_all.npy', Total_4span) 
np.save('./generated_MDOF_systems/Example_Modal_5span_all.npy', Total_5span) 
np.save('./generated_MDOF_systems/Example_Modal_6span_all.npy', Total_6span) 


#%% Estimate key modal values for predicting peak acceleration, see Eq. (17) of the reference
# Gamma_all, eigen_vect, s 
tmp_results_3span = MDOF_3span_modal_accel(p3_value)
tmp_results_4span = MDOF_4span_modal_accel(p4_value)
tmp_results_5span = MDOF_5span_modal_accel(p5_value)
tmp_results_6span = MDOF_6span_modal_accel(p6_value)

# Calculate values that required
val_3span = cal_value(tmp_results_3span[0], tmp_results_3span[1], tmp_results_3span[2])
val_4span = cal_value(tmp_results_4span[0], tmp_results_4span[1], tmp_results_4span[2])
val_5span = cal_value(tmp_results_5span[0], tmp_results_5span[1], tmp_results_5span[2])
val_6span = cal_value(tmp_results_6span[0], tmp_results_6span[1], tmp_results_6span[2])

Total_3span=(np.r_[val_3span[2], val_3span[6], val_3span[10]])
Total_4span=(np.r_[val_4span[2], val_4span[6], val_4span[10], val_4span[14]])
Total_5span=(np.r_[val_5span[2], val_5span[6], val_5span[10], val_5span[14], val_5span[18]])
Total_6span=(np.r_[val_6span[2], val_6span[6], val_6span[10], val_6span[14], val_6span[18], val_6span[22]])

       
np.save('./generated_MDOF_systems/Example_Modal_3span_accel.npy', Total_3span) 
np.save('./generated_MDOF_systems/Example_Modal_4span_accel.npy', Total_4span) 
np.save('./generated_MDOF_systems/Example_Modal_5span_accel.npy', Total_5span) 
np.save('./generated_MDOF_systems/Example_Modal_6span_accel.npy', Total_6span) 

