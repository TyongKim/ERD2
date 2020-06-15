"""

This script is able to predict the structural responses using the trained DNN
model. Please note that the DNN model is saved at 'DNN_model_2019.h5'.

Althought this script shows the predicted responses using a sample earthquake 
scenario, any kind of earthquake scenarios can be employed.

The deep learning model is developed based on the TensorFlow and Keras. Please
download the libraries.

The code is developed by Taeyong Kim from the Seoul National University
chs5566@snu.ac.kr

"""

# Basic libraries
import numpy as np
import matplotlib.pyplot as plt

#%% Fetch hysteretic system
###############################################################################
# Import hysteresis behaviors                                                 #
# So far we only use the predefined structural system                         #
# I will update this part that can use a general structural system, ASAP      #
# three hysteretic behaviros are employed                                     #
# HM1: linear hysteretic system 0~89                                          #
# HM2: Bilinear hysteretic system 90~27089                                    #
# HM3: Bilinear hysteretic w/ stiffness degradation system 27090~54089        #
# Period, 0.05 ~ 10 sec (90 steps)                                            #
# Yield force, 0.05 ~ 1.5 g (30 steps)                                        #
# Post yield stiffness ratio, 0~0.5 (10 steps)                                #
###############################################################################
# Structural index and features 
# The output is Index, period, stiffness, yield force (g), post-yield stiffness ratio
structural_index = np.load('structural_index.npy')   

# Hysteretic behaviors for DNN model - obtained by quasi-cyclic analysis
# The idicies fo structural index and hysteresis are matched each ohter
hysteresis = np.transpose(np.load('plot_hysteretic_data.npy'))

Hysteretic_ElemForc = hysteresis[0,1,:]  # Force
Hysteretic_NodeDisp = hysteresis[0,0,:]  # Displacement
Hysteretic_all=[]
for ii in range(54090):
    tmp = np.transpose(np.asarray([Hysteretic_NodeDisp[:,ii], Hysteretic_ElemForc[:,ii]]))
    Hysteretic_all.append(tmp)

Hysteretic_all = np.asarray(Hysteretic_all)
Hysteretic_all = Hysteretic_all.reshape(54090,80,2,1)

del hysteresis, Hysteretic_ElemForc, Hysteretic_NodeDisp, tmp, ii

# Total 25 structural system are employed
# HM2 with period (0.2, 0.5, 0.7, 1.0, 2.0s), yield force (0.1, 0.2, 0.3, 0.5, 1.0 g), post-yield stiffness ratio (0.05)
# Total of 25 structural systems
hys_index_target = [6151,6153,6155,6159,6169,
                    6721,6723,6725,6729,6739,
                    6871,6873,6875,6879,6889,
                    7051,7053,7055,7059,7069,
                    7351,7353,7355,7359,7369]

Target_structural_index = structural_index[hys_index_target,:]
Target_Hysteretic_input = Hysteretic_all[hys_index_target,:]
#%% Fetch Ground motion information
###############################################################################
# Import ground motion information                                            #
# You may need three different types of seismic information                   #
# 1. PGA (g), PGV (cm/s), PGD (cm)                                            #
# 2. Magnitude, Epicenter distance (km), Soil type                            #
# 3. Response sepctrum (110 steps from 0.005 ~ 10 sec)                        #
###############################################################################

# As a sample ground motions, we use 1970_Wrightwood_6074_Park_Dr_North ground motions
# The name of the file is '1970_Wrightwood_6074_Park_Dr_North.AT2'
# You can download the file at NGA-West2 database (https://ngawest2.berkeley.edu/)

# 1. Peak value information (PGA, PGV, PGD)
Input_GM_peak_info = [0.1445381, 8.512941, 1.251468]

# 2. Earthquake information
Input_earthquake_info = [5.33,	12.14, 0, 0 ,1 ,0 ,0] # Magnitude, epicenter distnace, and soil type (C type)

# 3. Response spectrum
from Newmark_TK import Newmark_TK # Newmark method that developed by Taeyong Kim
RS_period = np.load('Period.npy') # The target period (110 steps)
GM_acc = np.loadtxt('EX_GM0.txt') # accelerogram that changed from at2 --> txt
GM_time = np.loadtxt('EX_time0.txt') # Time step following the accelerogram

Input_RS_value = []
for ii in range(len(RS_period)):
    period = RS_period[ii]
    w = 2*np.pi/period        
    results = Newmark_TK(1, w**2, 0.05, GM_acc, GM_time, GM_time, 1/2, 1/6)
    Input_RS_value.append(np.max(np.abs(results[:,2])))

Input_RS_value = np.asarray(Input_RS_value)
Input_RS_value[0] = np.max(np.abs(GM_acc)) # this is to compensate numerical issue when period is very small
# plt.plot(RS_period, RS_value) # check the RS values by plotting

# Augment the ground motion information as the number of target structural systems
Target_Input_RS_value = []
Target_Input_earthquake_info  = []
Target_Input_GM_peak_info = []
for ii in range(len(Target_Hysteretic_input)):
    Target_Input_RS_value.append(np.log(Input_RS_value))  # Natural logarithm is employed 
    Target_Input_earthquake_info.append(Input_earthquake_info)
    Target_Input_GM_peak_info.append(np.log(Input_GM_peak_info)) # Natural logarithm is employed 

Target_Input_RS_value = np.asarray(Target_Input_RS_value)    
Target_Input_earthquake_info = np.asarray(Target_Input_earthquake_info)    
Target_Input_GM_peak_info = np.asarray(Target_Input_GM_peak_info)    
#%% Load trained DNN model
# The following line will help when using Mac OS
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from keras.models import load_model
model_disp = load_model('DNN_model_2019_displ.h5')
model_velo = load_model('DNN_model_2019_velo.h5')
model_acce = load_model('DNN_model_2019_accel.h5')
#%% predict the results by DNN model 
###############################################################################
# The DNN model produces the natural logarithm of the peak displacements      #
# velocity, and acceleration whose unit is 'm', 'm/s' and 'g', respectively.  #
# Thus, exponential function is applied to make the origianl unit.            #
###############################################################################
y_pred_disp  = model_disp.predict([Target_Hysteretic_input, Target_Input_GM_peak_info, Target_Input_earthquake_info, Target_Input_RS_value])
y_pred_velo  = model_velo.predict([Target_Hysteretic_input, Target_Input_GM_peak_info, Target_Input_earthquake_info, Target_Input_RS_value])
y_pred_acce  = model_acce.predict([Target_Hysteretic_input, Target_Input_GM_peak_info, Target_Input_earthquake_info, Target_Input_RS_value])

y_pred_disp = np.exp(y_pred_disp) # Unit m
y_pred_velo = np.exp(y_pred_velo) # Unit m/s
y_pred_acce = np.exp(y_pred_acce) # Unit m/s^2
