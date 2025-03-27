"""
This script preprocesses the data to serve as input for the DNN model.

Written by Taeyong Kim at Ajou University
taeyongkim@ajou.ac.kr
"""

import numpy as np
#%% Load data from structural analysis
# Ground motion information
Ground_info = np.load('./Ground_info/Ground_info_bridge.npy',allow_pickle=True) 

# Structural information
Structural_info_3span = np.load('./generated_MDOF_systems/Modal_3span_all.npy') 
Structural_info_4span = np.load('./generated_MDOF_systems/Modal_4span_all.npy') 
Structural_info_5span = np.load('./generated_MDOF_systems/Modal_5span_all.npy') 
Structural_info_6span = np.load('./generated_MDOF_systems/Modal_6span_all.npy') 

# Structural information for Acceleration
Results_str_accel_3span = np.load('./generated_MDOF_systems/Modal_3span_accel.npy') 
Results_str_accel_4span = np.load('./generated_MDOF_systems/Modal_4span_accel.npy') 
Results_str_accel_5span = np.load('./generated_MDOF_systems/Modal_5span_accel.npy') 
Results_str_accel_6span = np.load('./generated_MDOF_systems/Modal_6span_accel.npy') 

# MDOF results (displ)
Results_true_3span_displ = np.load('./Results_Summary_MDOF/Results_3span_displ.npy')
Results_true_4span_displ = np.load('./Results_Summary_MDOF/Results_4span_displ.npy')
Results_true_5span_displ = np.load('./Results_Summary_MDOF/Results_5span_displ.npy')
Results_true_6span_displ = np.load('./Results_Summary_MDOF/Results_6span_displ.npy')

# MDOF results (accel)
Results_true_3span_accel = np.load('./Results_Summary_MDOF/Results_3span_accel.npy')
Results_true_4span_accel = np.load('./Results_Summary_MDOF/Results_4span_accel.npy')
Results_true_5span_accel = np.load('./Results_Summary_MDOF/Results_5span_accel.npy')
Results_true_6span_accel = np.load('./Results_Summary_MDOF/Results_6span_accel.npy')

# Modal results (displ)
Results_modal_3span_displ = np.load('./Results_Summary_SDOF/Results_max_displ_3span.npy')
Results_modal_4span_displ = np.load('./Results_Summary_SDOF/Results_max_displ_4span.npy')
Results_modal_5span_displ = np.load('./Results_Summary_SDOF/Results_max_displ_5span.npy')
Results_modal_6span_displ = np.load('./Results_Summary_SDOF/Results_max_displ_6span.npy')

# Modal results (accel)
Results_modal_3span_accel = np.load('./Results_Summary_SDOF/Results_max_accel_3span.npy')
Results_modal_4span_accel = np.load('./Results_Summary_SDOF/Results_max_accel_4span.npy')
Results_modal_5span_accel = np.load('./Results_Summary_SDOF/Results_max_accel_5span.npy')
Results_modal_6span_accel = np.load('./Results_Summary_SDOF/Results_max_accel_6span.npy')

# Indicator for GM (Modal direction)
# 1: first mode transverse (y), 0: First mode longitudinal (x)
Results_gm_direction_3span = np.load('./Results_Summary_SDOF/Results_gm_direct_3span.npy')
Results_gm_direction_4span = np.load('./Results_Summary_SDOF/Results_gm_direct_4span.npy')
Results_gm_direction_5span = np.load('./Results_Summary_SDOF/Results_gm_direct_5span.npy')
Results_gm_direction_6span = np.load('./Results_Summary_SDOF/Results_gm_direct_6span.npy')

#%% Make input and output for the DNN model

DNN_input_GM_ver1 = []; DNN_input_accel_GM = []
DNN_input_STR = []; DNN_input_accel_STR = []
DNN_results_MDOF_displ = []; DNN_results_MDOF_accel = [] 
DNN_results_SDOF_displ = []; DNN_results_SDOF_accel = [];

idx = 0
for ii in range(3): # For each Ground motion

    tmp_Groud_info = Ground_info[ii]

    # (1) Ground motion intensity measure    
    # accel1 --> longitudinal (x), accel2 --> transverse (y)
    tmp2_GM_ver1_first = np.r_[np.log(tmp_Groud_info['Spec_value1']),
                               np.log(tmp_Groud_info['Spec_value2'])] # Use log to address the skewness
    
    tmp2_GM_ver1_accel = tmp_Groud_info['Peak_value2'] # PGA for accel
    
    for jj in range(2): # For each structure

        # (1)
        for kk in range(3): # 3span
            DNN_input_GM_ver1.append(tmp2_GM_ver1_first)
            DNN_input_accel_GM.append(tmp2_GM_ver1_accel)
        for kk in range(4): # 4span
            DNN_input_GM_ver1.append(tmp2_GM_ver1_first)
            DNN_input_accel_GM.append(tmp2_GM_ver1_accel)
                       
        for kk in range(5): # 5span
            DNN_input_GM_ver1.append(tmp2_GM_ver1_first)
            DNN_input_accel_GM.append(tmp2_GM_ver1_accel)

        for kk in range(6): # 6span
            DNN_input_GM_ver1.append(tmp2_GM_ver1_first)
            DNN_input_accel_GM.append(tmp2_GM_ver1_accel)
                
            
        # (2) Structural responses
        tmp_str_3span = Structural_info_3span[jj,:]
        tmp_str_4span = Structural_info_4span[jj,:]
        tmp_str_5span = Structural_info_5span[jj,:]
        tmp_str_6span = Structural_info_6span[jj,:]             
        
        # (2) Structural responses for accel responses
        tmp_accel_3span = Results_str_accel_3span[jj,:]
        tmp_accel_4span = Results_str_accel_4span[jj,:]
        tmp_accel_5span = Results_str_accel_5span[jj,:]
        tmp_accel_6span = Results_str_accel_6span[jj,:]        
        
        # (2) Convernt to DNN input for structural responses
        DNN_input_STR.append(np.r_[np.log(tmp_str_3span), 3, 1/3])
        DNN_input_STR.append(np.r_[np.log(tmp_str_3span), 3, 3/3])
        DNN_input_STR.append(np.r_[np.log(tmp_str_3span), 3, 1/3])
        
        DNN_input_STR.append(np.r_[np.log(tmp_str_4span), 4, 1/4])
        DNN_input_STR.append(np.r_[np.log(tmp_str_4span), 4, 3/4])
        DNN_input_STR.append(np.r_[np.log(tmp_str_4span), 4, 3/4])
        DNN_input_STR.append(np.r_[np.log(tmp_str_4span), 4, 1/4])
        
        DNN_input_STR.append(np.r_[np.log(tmp_str_5span), 5, 1/5])
        DNN_input_STR.append(np.r_[np.log(tmp_str_5span), 5, 3/5])        
        DNN_input_STR.append(np.r_[np.log(tmp_str_5span), 5, 5/5])        
        DNN_input_STR.append(np.r_[np.log(tmp_str_5span), 5, 3/5])        
        DNN_input_STR.append(np.r_[np.log(tmp_str_5span), 5, 1/5])
        
        DNN_input_STR.append(np.r_[np.log(tmp_str_6span), 6, 1/6])
        DNN_input_STR.append(np.r_[np.log(tmp_str_6span), 6, 3/6])        
        DNN_input_STR.append(np.r_[np.log(tmp_str_6span), 6, 5/6])        
        DNN_input_STR.append(np.r_[np.log(tmp_str_6span), 6, 5/6])        
        DNN_input_STR.append(np.r_[np.log(tmp_str_6span), 6, 3/6])        
        DNN_input_STR.append(np.r_[np.log(tmp_str_6span), 6, 1/6])        
        
        # (2) Convert to DNN input for structural acceleration responses
        DNN_input_accel_STR.append(np.r_[tmp_accel_3span, tmp_accel_4span,
                                         tmp_accel_5span, tmp_accel_6span])        
        
        # (3) True responses, displ
        tmp_MDOF_3span = Results_true_3span_displ[idx,:]
        tmp_MDOF_4span = Results_true_4span_displ[idx,:]
        tmp_MDOF_5span = Results_true_5span_displ[idx,:]
        tmp_MDOF_6span = Results_true_6span_displ[idx,:]
        
        # Conver to DNN input        
        DNN_results_MDOF_displ.append(np.r_[tmp_MDOF_3span, tmp_MDOF_4span,
                                      tmp_MDOF_5span, tmp_MDOF_6span])
        
        # (3) True responses, accel
        tmp_MDOF_3span = Results_true_3span_accel[idx,:]
        tmp_MDOF_4span = Results_true_4span_accel[idx,:]
        tmp_MDOF_5span = Results_true_5span_accel[idx,:]
        tmp_MDOF_6span = Results_true_6span_accel[idx,:]
        
        # Conver to DNN input        
        DNN_results_MDOF_accel.append(np.r_[tmp_MDOF_3span, tmp_MDOF_4span,
                                      tmp_MDOF_5span, tmp_MDOF_6span])
        
        
        # (4) Modal responses
        tmp_gm_direction_3span = Results_gm_direction_3span[idx]
        tmp_gm_direction_4span = Results_gm_direction_4span[idx]
        tmp_gm_direction_5span = Results_gm_direction_5span[idx]
        tmp_gm_direction_6span = Results_gm_direction_6span[idx]
                
        if tmp_gm_direction_3span ==0:
            tmp_SDOF_3span_displ = Results_modal_3span_displ[idx,:]
            tmp_SDOF_3span_accel = Results_modal_3span_accel[idx,:]
            
        else:
            tmp_SDOF_3span_displ = Results_modal_3span_displ[idx,:]
            tmp1 = np.array(tmp_SDOF_3span_displ[:,0])
            tmp2 = np.array(tmp_SDOF_3span_displ[:,1])
            tmp_SDOF_3span_displ[:,0] = tmp2
            tmp_SDOF_3span_displ[:,1] = tmp1

            tmp_SDOF_3span_accel = Results_modal_3span_accel[idx,:]
            tmp1 = np.array(tmp_SDOF_3span_accel[:,0])
            tmp2 = np.array(tmp_SDOF_3span_accel[:,1])
            tmp_SDOF_3span_accel[:,0] = tmp2
            tmp_SDOF_3span_accel[:,1] = tmp1            
        
        if tmp_gm_direction_4span ==0:
            tmp_SDOF_4span_displ = Results_modal_4span_displ[idx,:]
            tmp_SDOF_4span_accel = Results_modal_4span_accel[idx,:]
            
        else:
            tmp_SDOF_4span_displ = Results_modal_4span_displ[idx,:]
            tmp1 = np.array(tmp_SDOF_4span_displ[:,0])
            tmp2 = np.array(tmp_SDOF_4span_displ[:,1])
            tmp_SDOF_4span_displ[:,0] = tmp2
            tmp_SDOF_4span_displ[:,1] = tmp1

            tmp_SDOF_4span_accel = Results_modal_4span_accel[idx,:]
            tmp1 = np.array(tmp_SDOF_4span_accel[:,0])
            tmp2 = np.array(tmp_SDOF_4span_accel[:,1])
            tmp_SDOF_4span_accel[:,0] = tmp2
            tmp_SDOF_4span_accel[:,1] = tmp1          

        if tmp_gm_direction_5span ==0:
            tmp_SDOF_5span_displ = Results_modal_5span_displ[idx,:]
            tmp_SDOF_5span_accel = Results_modal_5span_accel[idx,:]
            
        else:
            tmp_SDOF_5span_displ = Results_modal_5span_displ[idx,:]
            tmp1 = np.array(tmp_SDOF_5span_displ[:,0])
            tmp2 = np.array(tmp_SDOF_5span_displ[:,1])
            tmp_SDOF_5span_displ[:,0] = tmp2
            tmp_SDOF_5span_displ[:,1] = tmp1

            tmp_SDOF_5span_accel = Results_modal_5span_accel[idx,:]
            tmp1 = np.array(tmp_SDOF_5span_accel[:,0])
            tmp2 = np.array(tmp_SDOF_5span_accel[:,1])
            tmp_SDOF_5span_accel[:,0] = tmp2
            tmp_SDOF_5span_accel[:,1] = tmp1  
            
        if tmp_gm_direction_6span ==0:
            tmp_SDOF_6span_displ = Results_modal_6span_displ[idx,:]
            tmp_SDOF_6span_accel = Results_modal_6span_accel[idx,:]
            
        else:
            tmp_SDOF_6span_displ = Results_modal_6span_displ[idx,:]
            tmp1 = np.array(tmp_SDOF_6span_displ[:,0])
            tmp2 = np.array(tmp_SDOF_6span_displ[:,1])
            tmp_SDOF_6span_displ[:,0] = tmp2
            tmp_SDOF_6span_displ[:,1] = tmp1

            tmp_SDOF_6span_accel = Results_modal_6span_accel[idx,:]
            tmp1 = np.array(tmp_SDOF_6span_accel[:,0])
            tmp2 = np.array(tmp_SDOF_6span_accel[:,1])
            tmp_SDOF_6span_accel[:,0] = tmp2
            tmp_SDOF_6span_accel[:,1] = tmp1  
            
            
        DNN_results_SDOF_displ.append(np.r_[np.c_[tmp_SDOF_3span_displ,np.zeros([3,12])],
                                            np.c_[tmp_SDOF_4span_displ,np.zeros([4,8])],
                                            np.c_[tmp_SDOF_5span_displ,np.zeros([5,4])],
                                            tmp_SDOF_6span_displ])        
        
        DNN_results_SDOF_accel.append(np.r_[np.c_[tmp_SDOF_3span_accel,np.zeros([3,12])],
                                            np.c_[tmp_SDOF_4span_accel,np.zeros([4,8])],
                                            np.c_[tmp_SDOF_5span_accel,np.zeros([5,4])],
                                            tmp_SDOF_6span_accel])        
                      

        idx = idx+1


DNN_input_GM_ver1 = np.asarray(DNN_input_GM_ver1)        
DNN_input_accel_GM = np.asarray(DNN_input_accel_GM)
DNN_input_STR = np.asarray(DNN_input_STR)
DNN_input_accel_STR = np.asarray(DNN_input_accel_STR)
DNN_results_MDOF_displ = np.asarray(DNN_results_MDOF_displ)        
DNN_results_MDOF_accel = np.asarray(DNN_results_MDOF_accel)
DNN_results_SDOF_displ = np.asarray(DNN_results_SDOF_displ)
DNN_results_SDOF_accel = np.asarray(DNN_results_SDOF_accel)


DNN_input_accel_STR2 = np.zeros([len(DNN_input_accel_GM),1])
DNN_results_MDOF_displ2 = np.zeros([len(DNN_input_accel_GM),1])
DNN_results_MDOF_accel2 = np.zeros([len(DNN_input_accel_GM),1])
DNN_results_SDOF_displ2= np.zeros([len(DNN_input_accel_GM),24])
DNN_results_SDOF_accel2 = np.zeros([len(DNN_input_accel_GM),24])

for ii in range(len(DNN_input_accel_STR)):

    tmp_DNN_input_accel_STR2 = DNN_input_accel_STR[ii,:]
    tmp_DNN_results_MDOF_displ2 = DNN_results_MDOF_displ[ii,:]
    tmp_DNN_results_MDOF_accel2 = DNN_results_MDOF_accel[ii,:]
    tmp_DNN_results_SDOF_displ2 = DNN_results_SDOF_displ[ii,:]
    tmp_DNN_results_SDOF_accel2 = DNN_results_SDOF_accel[ii,:]
    
    DNN_input_accel_STR2[ii*18:(ii+1)*18,:] = tmp_DNN_input_accel_STR2.reshape(18,1)
    DNN_results_MDOF_displ2[ii*18:(ii+1)*18,:] = tmp_DNN_results_MDOF_displ2.reshape(18,1)
    DNN_results_MDOF_accel2[ii*18:(ii+1)*18,:] = tmp_DNN_results_MDOF_accel2.reshape(18,1)
    DNN_results_SDOF_displ2[ii*18:(ii+1)*18,:] = tmp_DNN_results_SDOF_displ2.reshape(18,24)
    DNN_results_SDOF_accel2[ii*18:(ii+1)*18,:] = tmp_DNN_results_SDOF_accel2.reshape(18,24)
    

np.save('./Results_DL/DL_info_GM.npy', DNN_input_GM_ver1)
np.save('./Results_DL/DL_info_Accel_GM.npy', DNN_input_accel_GM)
np.save('./Results_DL/DL_info_Displ_str.npy', DNN_input_STR)
np.save('./Results_DL/DL_info_Accel_str.npy', DNN_input_accel_STR2)

np.save('./Results_DL/DL_info_MDOF_displ.npy', DNN_results_MDOF_displ2)
np.save('./Results_DL/DL_info_MDOF_accel.npy', DNN_results_MDOF_accel2)
np.save('./Results_DL/DL_info_SDOF_displ.npy', DNN_results_SDOF_displ2)
np.save('./Results_DL/DL_info_SDOF_accel.npy', DNN_results_SDOF_accel2)


