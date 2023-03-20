"""
This script is able to predict the structural responses using the trained DNN
model (ERD net). Please note that the DNN model is saved at 'ERD_net_displ.h5'.

Althought this script shows the predicted responses using a sample earthquake 
scenario, any kind of earthquake scenarios can be employed.

The deep learning model is developed based on the TensorFlow and Keras. Please
download the libraries.

The code is developed by Taeyong Kim at Ajou University
taeyongkim@ajou.ac.kr

"""

# Basic libraries
import numpy as np
import sqlite3
import tensorflow as tf
from tensorflow import keras
#%% Fetch hysteretic system
###############################################################################
# Import hysteresis behaviors                                                 #
# So far we only use the predefined structural system                         #
# Hysteresis generated using an m-BWBN model is employed                      #
###############################################################################
# Structural information from the database
# Only 100 systems among 129600 are employed
str_index = np.arange(100,200)
str_index_tmp2 = ', '.join(str(x) for x in str_index)

conn1 = sqlite3.connect("./Hysteretic_BWBN.db") # Should be changed
db1 = conn1.cursor() # Hysteretic model

Query_String = "select * "\
               "FROM Hys_behavior_displ "\
               "WHERE Random_num in (" + str(1) + ") and System_index in (" + str_index_tmp2 + ") "\
               "    order by System_index "\

db1.execute(Query_String) 
input_db  = db1.fetchall()
Displacement = np.asarray(input_db)    

Query_String = "select * "\
               "FROM Hys_behavior_force "\
               "WHERE Random_num in (" + str(1) + ") and System_index in (" + str_index_tmp2 + ") "\
               "    order by System_index "\

db1.execute(Query_String) 
input_db  = db1.fetchall()
Force = np.asarray(input_db)      

Hysteretic = []
for ii in range(len(Force)):
    tmp_displ = np.r_[0, Displacement[ii,2:-1]]
    tmp_force = np.r_[0, Force[ii,2:-1]]
    
    tmp_Hysteretic = np.c_[tmp_displ, tmp_force]

    Hysteretic.append(tmp_Hysteretic)
    
Hysteretic = np.asarray(Hysteretic)
Hysteretic = Hysteretic.reshape(len(Force),241,2,1)

#%% Fetch Ground motion information
###############################################################################
# Import ground motion information                                            #
# You may need three different types of seismic information                   #
# 1. PGA (g), PGV (cm/s), PGD (cm), zero_crossing, Arias intensity, Housner   #
# 2. Magnitude, Epicenter distance (km), Soil type                            #
# 3. Response sepctrum (110 steps from 0.005 ~ 10 sec)                        #
###############################################################################
# An instance of ground motion information

# 1. Ground motion information
tmp_P_info = [0.149772  ,  5.7720345 ,  1.0334056 , 25. ,  0.44517468,
             15.64293197] # PGA, PGV, PGD, zero_corssing, Arias, Housner intensites

# 2. Earthquake information
tmp_EQ_info = [6.  , 6.31, 0.  , 0.  , 1.  , 0.  , 0.  ] # C class soil type (0,0,1,0,0)

# 3. Response spectrum
tmp_Sa_info = [0.15173679, 0.15299831, 0.15498599, 0.15619643, 0.16142953, 0.16976556,
               0.1856035, 0.20976493, 0.22154328, 0.25328007, 0.28370661, 0.33179421, 
               0.36055452, 0.3888935, 0.38508434, 0.37795492, 0.32817695,  0.30479899, 
               0.29691999, 0.33545198, 0.45614778, 0.39036372, 0.4069315, 0.4971994, 
               0.44882944, 0.33955584, 0.27617806,  0.2164001, 0.1837714,
               0.16776384, 0.18803145, 0.22334573, 0.21642768,
               0.20751573, 0.22049974,  0.24001313, 0.24618188,
               0.23863686, 0.22080058, 0.19726251, 0.17406945,
               0.16616209, 0.18190558, 0.213826, 0.24571753,
               0.26885486, 0.27782005, 0.27165371, 0.25574394,
               0.24628809, 0.22461993, 0.19719759, 0.18421857,
               0.17792155, 0.17232829, 0.16539663, 0.15838063,
               0.15508324, 0.15149825,  0.14817957, 0.12997372,
               0.11045063, 0.091461089, 0.074262903, 0.059016636,
               0.04615128, 0.038468127, 0.032328012, 0.027974144,
               0.024379892, 0.026199534, 0.025093232, 0.025479612,
               0.024785751, 0.024258064, 0.022893814, 0.021838467,
               0.020611772, 0.019391346, 0.01791911, 0.016369236,
               0.014792358, 0.013296033, 0.012011, 0.011159103,
               0.010303733, 0.0093347119, 0.0084094427, 0.0075595479,
               0.0068917861, 0.005841258, 0.0049155545, 0.0043950349,
               0.0038761835, 0.0035060025, 0.0031978668, 0.0029249984,
               0.0026831384, 0.0024684534, 0.0022774996, 0.0018850608,
               0.0015888125, 0.0013599065, 0.0011792859, 0.0010344686,
               0.00091664233, 0.00081948962, 0.00073840964, 0.00066999986,
               0.00061170674]

EQ_info = []
P_info = []
Sa_info = []
for ii in range(len(str_index)):
    EQ_info.append(tmp_EQ_info)
    P_info.append(np.log(tmp_P_info))
    Sa_info.append(np.log(tmp_Sa_info))

EQ_info = np.asarray(EQ_info)
P_info = np.asarray(P_info)
Sa_info = np.asarray(Sa_info)
#%% Load trained DNN model
model = keras.models.load_model('ERD_net_displ.h5')

#%% predict the results by DNN model 
Predicted_results = model.predict([Hysteretic, EQ_info, Sa_info, P_info])
Predicted_results = np.exp(Predicted_results)  # Unit m
