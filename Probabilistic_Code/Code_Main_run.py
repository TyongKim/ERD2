"""
After running 'Code_Main_script.py', this script predict a structural 
responses (Peak displacement, velocity, and acceleration) using the P-DNN model 
developed by Kim et al. (2020).
Note that 'Code_Main_script.py' contains the hysteretic and ground motion
information of interest.

Developed by Taeyong Kim from the Seoul National Universtiy
chs5566@snu.ac.kr
December 10, 2020

Reference
Kim, T., Song, J., and Kwon, O.S. (2020). Probabilistic evaluation of seismic 
responses using deep learning method. Structural Safety. Vol. 84, 101913.
"""

# Basic libraries
import numpy as np
from scipy import interpolate
#%% Preprocessing of hysteretic behaviors
###############################################################################
# Find the index of the defined structural system                             #
# The results of the parameters are 'Character' and 'Index'                   #
# 'Character': summary of the perdefined structural characteristics           #
# 'Index': index of a structural system that corresponds to a pre-generated   #
#          hysteresis saved in 'plot_hysteretic_data.npy'                     #
###############################################################################

structural_index = np.load('structural_index.npy')   

period = structural_index[0:90,1]
yield_force = structural_index[90:120,3]
post_yield_stiff_index = np.arange(90,27090,2700)
post_yield_stiffness_ratio = structural_index[post_yield_stiff_index,4]

if Hysteretic_model[-1] == str(1):

    Hysteretic_period = input("Period (s): ")
    tmp = np.abs(float(Hyst_period) - period)
    Index_period = np.where(tmp == np.min(tmp))[0]
    if len(Index_period)>1:
        Index_period = Index_period[0]
        
    Character = Index_period
    Index = Index_period
    
elif Hysteretic_model[-1] == str(2) or Hysteretic_model[-1] == str(3):

    tmp = np.abs(float(Hyst_period) - period)
    Index_period = np.where(tmp == np.min(tmp))[0]   
    if len(Index_period)>1:
        Index_period = Index_period[0]
        
    tmp = np.abs(float(Hyst_yield_force) - yield_force)
    Index_yield = np.where(tmp == np.min(tmp))[0]
    if len(Index_yield)>1:
        Index_yield = Index_yield[0]        
    
    tmp = np.abs(float(Hyst_post_yield) - post_yield_stiffness_ratio)
    Index_post = np.where(tmp == np.min(tmp))[0] 
    if len(Index_post)>1:
        Index_post = Index_post[0]     
        
    Character = [period[int(Index_period)], 
                 yield_force[int(Index_yield)], post_yield_stiffness_ratio[int(Index_post)]]
    
    if Hysteretic_model[-1] == str(2):
        Index =  90 + Index_period*30 + Index_yield + Index_post*2700
    elif Hysteretic_model[-1] == str(3):
        Index =  27090 + Index_period*30 + Index_yield + Index_post*2700

# Find the hysteretic behaviors from the pre-caluclated hysteresis
hysteresis = np.transpose(np.load('plot_hysteretic_data.npy'))

Hysteretic_ElemForc = hysteresis[0,1,:]  # Force
Hysteretic_NodeDisp = hysteresis[0,0,:]  # Displacement
Hysteretic_all=[]
for ii in range(54090):
    tmp = np.transpose(np.asarray([Hysteretic_NodeDisp[:,ii], Hysteretic_ElemForc[:,ii]]))
    Hysteretic_all.append(tmp)

Hysteretic_all = np.asarray(Hysteretic_all)
Hysteretic_all = Hysteretic_all.reshape(54090,80,2,1)
        
Input_Hysteretic_input = Hysteretic_all[Index].reshape(1,80,2,1)
#%% Preprocessing of ground motions
###############################################################################
# Preprocessing of ground motions information                                 #
# Response spectrum is extended using the predefined period steps.            #
# Soil type is converted to numerical variables through one-hot encoding.     #        
# Natural logarithm is applied to response spectrum and peak values of ground #
# motions.                                                                    #
# Please note that the more coarse response spectrum value is provided,       #
# the more precise responses can be obtained.                                 #
###############################################################################

# Interpolate the response spectrum
Period = np.load('Period.npy')  # predefined period steps (110x1)
peri = np.zeros([np.size(Input_RS_value,0),2])
for ii in range(23):
    peri[ii,0] = Input_RS_value.iloc[ii,0]
    peri[ii,1] = Input_RS_value.iloc[ii,1]
    
interpolator_spectrum = interpolate.interp1d(peri[:,0], peri[:,1], kind = 'linear')
Input_RS_value2 = interpolator_spectrum(Period)
Input_RS_value2 = np.log(Input_RS_value2) # Natural logarithm
Input_RS_value2 = Input_RS_value2.reshape(1,110)

# Peak values
Input_GM_peak_info2 = np.asarray(np.log(Input_GM_peak_info))
Input_GM_peak_info2 = Input_GM_peak_info2.reshape(1,3)

# One-hot encoding
# if no specific site information is provided, we assume 'C' class
temp_site = Input_earthquake_info[2]
if temp_site == 'A':
    site = 0
elif temp_site =='B':
    site = 1
elif temp_site =='C':
    site = 2
elif temp_site =='D':
    site = 3
elif temp_site =='E':
    site = 4
else:
    site = 2
   
Site = np.eye(5)[site]
Input_earthquake_info2 = np.r_[Input_earthquake_info[0:2], Site]
Input_earthquake_info2 = Input_earthquake_info2.reshape(1,7)
#%% Predict structural responses using DNN models
###############################################################################
# Predict structural responses using P-DNN model                              #
# Maximum transient displacement (m), velocity (m/s), and acceleration (m/s^2)#
###############################################################################
# The following line will help when using Mac OS
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Load deep neural network models
# Because the model was developed using customized loss function, the following
# procedure is requried.

import tensorflow as tf
import keras
# Define loss function which is written in Eq. (2) of the reference
def mse_lin_wrapper(var):

    def mse_lin(y_true, y_pred):
        return robust_mse(y_true, y_pred, var)
    
    # set the name to be displayed in TF/Keras log
    mse_lin.__name__ = 'mean_squared_error_prediction'
    
    return mse_lin

def mse_var_wrapper(lin):

    def mse_var(y_true, y_pred):
        return robust_mse(y_true, lin, y_pred)
    
    # set the name to be displayed in TF/Keras log
    mse_var.__name__ = 'mean_squared_error_predictive_variance'  

    return mse_var

def robust_mse(y_true, y_pred, variance):
    # Negative log likelihood of Gaussian distribution is set as the loss function.
    # Neural Net is predicting log(var), so take exp, takes into account 
    # the target variance, and take log back.
    # This is due to the numerical convergence during training.    
    y_pred_corrected = tf.math.log(tf.math.exp(variance))

    wrapper_output = (0.5 * tf.math.square(y_true - y_pred) 
                      * (tf.math.exp(-y_pred_corrected)) 
                      + 0.5 *y_pred_corrected)

    return tf.reduce_mean(wrapper_output, axis=-1)

from keras.layers import Input
from keras.layers import Dense

linear_loss = mse_var_wrapper(Dense(units=1, name="PGA_Sa_MRnew8_1", activation='linear')(Input(shape = (32,))))
aleato_loss = mse_lin_wrapper(Dense(units=1, name="PGA_Sa_MRnew8_2", activation='linear')(Input(shape = (32,))))

model_disp = keras.models.load_model('P_DNN_model_2019_displ.h5'
             , custom_objects = {'mean_squared_error_prediction':linear_loss, 'mean_squared_error_predictive_variance':aleato_loss})
model_velo = keras.models.load_model('P_DNN_model_2019_velo.h5'
             , custom_objects = {'mean_squared_error_prediction':linear_loss, 'mean_squared_error_predictive_variance':aleato_loss})
model_acce = keras.models.load_model('P_DNN_model_2019_accel.h5'
             , custom_objects = {'mean_squared_error_prediction':linear_loss, 'mean_squared_error_predictive_variance':aleato_loss})

# Predict the structural responses after loading the trained DNN model
y_val_disp, y_pred_disp = model_disp.predict([Input_Hysteretic_input, Input_GM_peak_info2, Input_earthquake_info2, Input_RS_value2])
y_val_velo, y_pred_velo = model_velo.predict([Input_Hysteretic_input, Input_GM_peak_info2, Input_earthquake_info2, Input_RS_value2])
y_val_acce, y_pred_acce = model_acce.predict([Input_Hysteretic_input, Input_GM_peak_info2, Input_earthquake_info2, Input_RS_value2])

y_pred_disp = np.exp(y_pred_disp) # Unit m
y_pred_velo = np.exp(y_pred_velo) # Unit m/s
y_pred_acce = np.exp(y_pred_acce) # Unit m/s^2

y_std_disp = np.sqrt(np.exp(y_val_disp))
y_std_velo = np.sqrt(np.exp(y_val_velo))
y_std_acce = np.sqrt(np.exp(y_val_acce))

#%% Print results
if Hysteretic_model[-1] == str(1):
    print('Structural system \n')
    print('Period (s):',Character,'\n')
    print('Peak displacement (m): ', y_pred_disp); print('Corresponding std.: ', y_std_disp)
    print('Peak velocity (m/s): ', y_pred_velo); print('Corresponding std.: ', y_std_velo)
    print('Peak acceleration (m/s^2): ', y_pred_acce); print('Corresponding std.: ', y_std_acce)    
else:
    print('Structural system')
    print('Period (s): ',Character[0])
    print('Yield stiffness ratio (g):  ',Character[1])
    print('Post-yield stiffness ratio: ',Character[2])
    print('Peak displacement (m): ', y_pred_disp); print('Corresponding std.: ', y_std_disp)
    print('Peak velocity (m/s): ', y_pred_velo); print('Corresponding std.: ', y_std_velo)
    print('Peak acceleration (m/s^2): ', y_pred_acce); print('Corresponding std.: ', y_std_acce)
