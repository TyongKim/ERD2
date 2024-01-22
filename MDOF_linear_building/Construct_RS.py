###############################################################################
# This code is developed by Prof. Taeyong Kim at Ajou University              #
# taeyongkim@ajou.ac.kr                                                       #
# Jan 20, 2024                                                                #
###############################################################################

# Import libraries
import numpy as np
import pandas as pd

#%% discretize SDOF systems
dp = 0.015
period_candidate = np.arange(np.log(0.05)-6*dp,np.log(4)+dp,dp)
period_candi = np.exp(period_candidate)

del dp

ddm = 0.08
damping_candidate = np.arange(np.log(0.005),np.log(0.25)+ddm,ddm) # 0.5% to 20%
damping_candi = np.exp(damping_candidate)
del ddm

#%%
num_GM = 2 # Number of ground motions used to construct the database
tmp_RS = np.zeros([num_GM, len(period_candi),len(damping_candi)])
for idx_gm in range(num_GM): # number of ground motions
    
    for idx_pe in range(len(period_candi)):
        tmp_direct_file = './Results_displ/GM_%d/period_%d.p' %(idx_gm,idx_pe)
        df = pd.read_pickle(tmp_direct_file)
        
        for idx_da in range(len(damping_candi)):
            tmp = df.iloc[:,idx_da+1].values
            tmp_RS[idx_gm, idx_pe,idx_da] = np.max(np.abs(tmp))


tmp_file_name = './Results_displ/A_Response_spectrum.npy'
np.save(tmp_file_name, tmp_RS )
