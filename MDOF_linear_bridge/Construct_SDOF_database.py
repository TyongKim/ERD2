"""
This script is to perform dynamic anlaysis for each mode and save peak 
displacement and acceleration values

Written by Taeyong Kim at Ajou University
taeyongkim@ajou.ac.kr
"""


import numpy as np
from i_MDOF_matrix_all import MDOF_3span_analysis

#%% 3-span
# Load the MDOF systems
p3_value = np.load('./generated_MDOF_systems/3_span.npy', allow_pickle=True)

Results_max_abs = []
Results_gm_direction = []
for ii in range(3): # Number of ground motions

    # GM name
    tmp1_1 = './Ground_motion/GM1_%d' %(ii)  # longitudinal
    tmp1_2 = './Ground_motion/GM2_%d' %(ii)  # transverse
    tmp2   = './Ground_motion/time_%d' %(ii)        
    
    gm_length = len(np.loadtxt(tmp1_1))
    gm_time_ratio = np.max([1, np.loadtxt(tmp2)/0.001])
    gm_timeseries = np.linspace(0,np.loadtxt(tmp2)*gm_length*1.3, int(gm_length*gm_time_ratio*1.3))
    
    tmp_Results_max_abs = []
    tmp_Results_gm_direction = []
    for jj in range(5): # for each structure

        # [mass_column, sitff_y_super, stiff_x_column, stiff_y_column, tmp_damping, random_values]
        tmp_value2 = p3_value[jj, :]   
        
        # SDOF analysis
        results = MDOF_3span_analysis(tmp_value2, tmp1_1, tmp1_2, tmp2)        
        results = np.asarray(results)
        dd
        
        # Save responses for each mode, only the center node of each span is saved   
        if np.max(np.abs(results[0,0,:])) == 0: # first mode transverse
            # 1st mode: out ouf plane
            results_max_abs = np.array([
                                   [np.max(np.abs(results[0,2,:])), np.max(np.abs(results[1,0,:])), np.max(np.abs(results[2,2,:])),
                                    np.max(np.abs(results[3,2,:])), np.max(np.abs(results[4,2,:])), np.max(np.abs(results[5,2,:])),
                                    np.max(np.abs(results[6,2,:])), np.max(np.abs(results[7,2,:])), np.max(np.abs(results[8,2,:])),
                                    np.max(np.abs(results[9,2,:])), np.max(np.abs(results[10,2,:])), np.max(np.abs(results[11,2,:]))], # Node 2
                                   
                                   [np.max(np.abs(results[0,6,:])), np.max(np.abs(results[1,0,:])), np.max(np.abs(results[2,6,:])),
                                    np.max(np.abs(results[3,6,:])), np.max(np.abs(results[4,6,:])), np.max(np.abs(results[5,6,:])),
                                    np.max(np.abs(results[6,6,:])), np.max(np.abs(results[7,6,:])), np.max(np.abs(results[8,6,:])),
                                    np.max(np.abs(results[9,6,:])), np.max(np.abs(results[10,6,:])), np.max(np.abs(results[11,6,:]))], # Node 6

                                   [np.max(np.abs(results[0,10,:])), np.max(np.abs(results[1,0,:])), np.max(np.abs(results[2,10,:])),
                                    np.max(np.abs(results[3,10,:])), np.max(np.abs(results[4,10,:])), np.max(np.abs(results[5,10,:])),
                                    np.max(np.abs(results[6,10,:])), np.max(np.abs(results[7,10,:])), np.max(np.abs(results[8,10,:])),
                                    np.max(np.abs(results[9,10,:])), np.max(np.abs(results[10,10,:])), np.max(np.abs(results[11,10,:]))], # Node 10
                                    

                                    ])
            tmp_Results_gm_direction.append(1) # out of plane
            
        else: # first mode longitudinal
            # 1st mode: longitudinal
            results_max_abs = np.array([
                                   [np.max(np.abs(results[0,0,:])), np.max(np.abs(results[1,2,:])), np.max(np.abs(results[2,2,:])),
                                    np.max(np.abs(results[3,2,:])), np.max(np.abs(results[4,2,:])), np.max(np.abs(results[5,2,:])),
                                    np.max(np.abs(results[6,2,:])), np.max(np.abs(results[7,2,:])), np.max(np.abs(results[8,2,:])),
                                    np.max(np.abs(results[9,2,:])), np.max(np.abs(results[10,2,:])), np.max(np.abs(results[11,2,:]))], # Node 2

                                   [np.max(np.abs(results[0,0,:])), np.max(np.abs(results[1,6,:])), np.max(np.abs(results[2,6,:])),
                                    np.max(np.abs(results[3,6,:])), np.max(np.abs(results[4,6,:])), np.max(np.abs(results[5,6,:])),
                                    np.max(np.abs(results[6,6,:])), np.max(np.abs(results[7,6,:])), np.max(np.abs(results[8,6,:])),
                                    np.max(np.abs(results[9,6,:])), np.max(np.abs(results[10,6,:])), np.max(np.abs(results[11,6,:]))], # Node 6

                                   [np.max(np.abs(results[0,0,:])), np.max(np.abs(results[1,10,:])), np.max(np.abs(results[2,10,:])),
                                    np.max(np.abs(results[3,10,:])), np.max(np.abs(results[4,10,:])), np.max(np.abs(results[5,10,:])),
                                    np.max(np.abs(results[6,10,:])), np.max(np.abs(results[7,10,:])), np.max(np.abs(results[8,10,:])),
                                    np.max(np.abs(results[9,10,:])), np.max(np.abs(results[10,10,:])), np.max(np.abs(results[11,10,:]))], # Node 10

                                    
                                    ])       
            tmp_Results_gm_direction.append(0) # longitudinal
            

        tmp_Results_max_abs.append(results_max_abs)

    Results_max_abs.append(tmp_Results_max_abs)

    print('next'); print(ii)
    
    
np.save('./Results_Summary_SDOF/Results_max_abs_3span_ver3_all.npy', Results_max_abs)



































