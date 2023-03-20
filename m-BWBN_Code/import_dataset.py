###############################################################################
# This script is to import dataset from databse of m-BWBN model.              #
# Total of 440 structural systems with 1,499 ground motions.                  #
# Developed by Taeyong Kim from the Seoul National University                 #
# July 30 2020 chs5566@snu.ac.kr                                              #
###############################################################################


#%% 
def Hysteretic_randomness(Force, Displacement):

    import numpy as np
    import scipy.ndimage

    Hysteretic = []
    for ii in range(len(Force)):
        tmp_displ = np.r_[0, Displacement[ii,2:-1]]
        tmp_force = np.r_[0, Force[ii,2:-1]]
        
        tmp_Hysteretic = np.c_[tmp_displ, tmp_force]                

        # Put some randomness   
        rand_shift_value = np.random.randint(-10,50) # 5 datasets
        tmp_Hysteretic = scipy.ndimage.shift(tmp_Hysteretic, [rand_shift_value,0],mode='reflect')
                        
        if np.random.rand()<0.5: # flip randomness
            tmp_Hysteretic = np.flipud(tmp_Hysteretic)

        #if np.random.rand()<0.5: # flip randomness
        #    tmp_Hysteretic = np.fliplr(tmp_Hysteretic)
                    
        if np.random.rand()<0.5: # flip randomness
            tmp_Hysteretic = -tmp_Hysteretic

        Hysteretic.append(tmp_Hysteretic)
        
    Hysteretic = np.asarray(Hysteretic)
    Hysteretic = Hysteretic.reshape(len(Force),241,2,1)
    
    return Hysteretic

def GM_randomness(tmp3):
    import numpy as np
    
    tmp_EQ = tmp3['Earthquake']
    tmp_P_new = np.r_[tmp3['Peak_value'], tmp3['zerocross_strong'],
                      tmp3['Arias_intensity'], tmp3['Housner']]
    tmp_Sa_new = tmp3['Spect_acce']
    
    tmp_EQ_new = tmp_EQ
    tmp_P_new = np.log(tmp_P_new)
    tmp_Sa_new = np.log(tmp_Sa_new)
        
    return [tmp_EQ_new, tmp_P_new, tmp_Sa_new]

#%%
def import_dataset(RSN_training, Num_train_GM):
    import numpy as np
    import sqlite3
    import random

    # Information of 1,499 GM from NGA West Database
    Gm_informat = np.load('./Ground_motions/Ground_info_final.npy',allow_pickle='TRUE')
    
    GM_RSN = []
    for ii in range(len(Gm_informat)):
        tmp2 = Gm_informat[ii]
        GM_RSN.append(tmp2['RSN'])    
    
    #%% BWBN model
    # Connect database - New database
    conn2 = sqlite3.connect("./SDDB_mBWBN_ver1.0.db")
    db2 = conn2.cursor() # Analysis results of dynamic analysis   
    
    conn1 = sqlite3.connect("./Hysteretic_BWBN.db") # Should be changed
    db1 = conn1.cursor() # Hysteretic model


    # Randomly select groud motions
    random.shuffle(RSN_training)
    RSN_training_original_modify = RSN_training[0:Num_train_GM] 
    
    # Fetch dataset that corresponds to RSN_training_original_modify
    Output_BWBN_dis = []
    X1_MR_BW = [];  X1_Sa_BW = [];  X1_P_BW = [];   
    for ii in range(Num_train_GM):
        str_index = np.arange(0,129600)
        
        RSN_training_original_modify_tmp = RSN_training_original_modify[ii]
        RSN_training_original_modify_tmp = np.asarray(RSN_training_original_modify_tmp)

        str_index_tmp2 = ', '.join(str(x) for x in str_index)

        # select RSN for training
        RSN_training = np.random.choice(RSN_training_original_modify_tmp, 1, replace=False)
        tmp = RSN_training[0]
        tmp = np.where(GM_RSN==tmp)[0]            
        tmp2 = ', '.join(str(x) for x in tmp)            
        
        # Fetch dataset of response dataset
        Query_String = "select \"Displacement(m)\", Ground_motion_id, Structural_system_id "\
                       "FROM Analysis_result "\
                       "WHERE Ground_motion_id in (" + tmp2 + ") and Structural_system_id in (" + str_index_tmp2 + ") "\
                       "    order by Structural_system_id "\
                           
        db2.execute(Query_String)
        input_db  = db2.fetchall()
        input_db = np.asarray(input_db)
        
        Dis_transient_total = input_db[:,0]
        Output_training = np.log(Dis_transient_total)
        Output_training = np.asarray(Output_training)
        
        if ii==0:
            Output_BWBN_dis = Output_training
        else:
            Output_BWBN_dis = np.r_[Output_BWBN_dis, Output_training] # Stack the variables
        
        # Fetch dataset of ground motions
        tmp3 = Gm_informat[int(tmp)]
        for kk in range(len(Output_training)):
            [tmp_EQ_new, tmp_P_new, tmp_Sa_new] = GM_randomness(tmp3)
            X1_MR_BW.append(tmp_EQ_new)
            X1_Sa_BW.append(tmp_Sa_new)
            X1_P_BW.append(tmp_P_new)
        
        # Fetch dataset of hysteretic systems
        Ran_num = 1 #np.random.randint(1,6) # randomly select from 1 to 5
        Ran_num2 = str(Ran_num)
  
        Query_String = "select * "\
                       "FROM Hys_behavior_displ "\
                       "WHERE Random_num in (" + Ran_num2 + ") and System_index in (" + str_index_tmp2 + ") "\
                       "    order by System_index "\
        
        db1.execute(Query_String) 
        input_db  = db1.fetchall()
        Displacement = np.asarray(input_db)    
        
        Query_String = "select * "\
                       "FROM Hys_behavior_force "\
                       "WHERE Random_num in (" + Ran_num2 + ") and System_index in (" + str_index_tmp2 + ") "\
                       "    order by System_index "\
        
        db1.execute(Query_String) 
        input_db  = db1.fetchall()
        Force = np.asarray(input_db)      
        
        Hys_train_BW = Hysteretic_randomness(Force, Displacement)
        if ii==0:
            X1_Hysteretic_BW = Hys_train_BW
        else:
            X1_Hysteretic_BW = np.r_[X1_Hysteretic_BW, Hys_train_BW]
    
    
    
    conn1.close()
    conn2.close()
    del db1, db2, conn1, conn2, input_db, Displacement, Force, str_index

    X1_MR_BW = np.asarray(X1_MR_BW)
    X1_Sa_BW = np.asarray(X1_Sa_BW)
    X1_P_BW = np.asarray(X1_P_BW)
    X1_Hysteretic_BW = np.asarray(X1_Hysteretic_BW)
    Output_BWBN_dis = np.asarray(Output_BWBN_dis)

    
    return [X1_MR_BW, X1_Sa_BW, X1_P_BW, X1_Hysteretic_BW, Output_BWBN_dis]

