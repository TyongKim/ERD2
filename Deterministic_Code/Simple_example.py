"""
Taeyong Kim at Seoul National University

This script presents a simple example for using the trained DNN model.It 
produces a displacement, velocity and acceleration for given inputs. One can 
freely modify the script.
"""

# import libraries
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

#%% Customized input

# Ground motion information
# Peak ground acceleration, velocity and displacement
PGA = 0.2137    #g
PGV = 1         #m/s
PGD = 1         #m

# Spectral acceleration
period = [0.005,0.01,0.02,0.03,0.04,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,
          0.75,1,1.5,2,3,4,5,7.5,10] # Sec

Sa = [0.2137,0.2137,0.2145,0.2238,0.2355,0.2493,0.2949,0.3377,0.4284,0.4796,
      0.4899,0.4812,0.4498,0.4074,0.3229,0.2687,0.1599,0.1035,0.0552,0.0349,
      0.0247,0.0117,0.0062] # g

# Magnitude, epicenter distnace, and soil type
M = 6.5      # Magnitude
R = 30       # Epicenter distance, km
temp_site = 'C'   # Site class, BSSC 2000: A,B,C,D and E

# Hysteretic behavior
# bilinear model Period = 0.065 sec, Fy = 0.05g, alpha =0
hysteretic_node = [2.20248e-05,5.244e-05,3.46104e-05,-1.4555e-20,-2.622e-05,
                   -5.244e-05,-3.46104e-05,3.78431e-20,2.622e-05,5.244e-05,
                   7.86601e-05,0.00010488,8.70505e-05,5.244e-05,2.622e-05,
                   -2.24148e-19,-2.622e-05,-5.244e-05,-7.86601e-05,-0.00010488,
                   -8.70505e-05,-5.244e-05,-2.622e-05,4.10452e-19,2.622e-05,
                   5.244e-05,7.86601e-05,0.00010488,0.0001311,0.00015732,
                   0.000139491,0.00010488,7.86601e-05,5.244e-05,2.622e-05,
                   -5.96756e-19,-2.622e-05,-5.244e-05,-7.86601e-05,-0.00010488,
                   -0.0001311,-0.00015732,-0.000139491,-0.00010488,-7.86601e-05,
                   -5.244e-05,-2.622e-05,9.69365e-19,2.622e-05,5.244e-05,
                   7.86601e-05,0.00010488,0.0001311,0.00015732,0.00018354,
                   0.00020976,0.000191931,0.00015732,0.0001311,0.00010488,
                   7.86601e-05,5.244e-05,2.622e-05,-1.15567e-18,-2.622e-05,
                   -5.244e-05,-7.86601e-05,-0.00010488,-0.0001311,-0.00015732,
                   -0.00018354,-0.00020976,-0.000191931,-0.00015732,-0.0001311,
                   -0.00010488,-7.86601e-05,-5.244e-05,-2.622e-05,1.34197e-18] #m

hysteretic_force = [0.021,0.05,0.033,-6.93889e-18,-0.025,-0.05,-0.033,
                    4.16334e-17,0.025,0.05,0.05,0.05,0.033,-2.98372e-16,-0.025,
                    -0.05,-0.05,-0.05,-0.05,-0.05,-0.033,6.03684e-16,0.025,
                    0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.033,-1.06165e-15,
                    -0.025,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,
                    -0.05,-0.033,1.51268e-15,0.025,0.05,0.05,0.05,0.05,0.05,
                    0.05,0.05,0.05,0.05,0.05,0.05,0.033,-1.8735e-15,-0.025,
                    -0.05,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,
                    -0.05,-0.05,-0.05,-0.05,-0.033,2.56739e-15,0.025,0.05,
                    0.05,0.05,0.05,0.05] #g

#%% Post process
# PGA, PGV, and PGD
X1_P = np.log([PGA, PGV, PGD]).reshape(1,3)

# Spectral accelration - make the vector 110X1
period_110 = [0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,
              0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.095,0.1,0.11,0.12,0.13,
              0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,
              0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,
              0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.55,0.6,
              0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7
              ,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3,3.2,3.4,3.6,3.8,
              4,4.2,4.4,4.6,4.8,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]


interpolator_spectrum = interpolate.interp1d(period, Sa, kind = 'linear')
X1_Sa = interpolator_spectrum(period_110)
X1_Sa = np.log(X1_Sa).reshape(1,110)

# Soil type
Site= []
if temp_site == 'A':
    Site.append(0)
elif temp_site =='B':
    Site.append(1)
elif temp_site =='C':
    Site.append(2)
elif temp_site =='D':
    Site.append(3)
elif temp_site =='E':
    Site.append(4)
else:
    Site.append(2)
    
targets = np.array(Site).reshape(-1)    
Site = np.eye(5)[targets]

X1_MR = np.c_[M, R, Site]

# Hysteretic information
Hysteretic_info_total = np.transpose(np.asarray([
                        hysteretic_node,hysteretic_force])).reshape(1,80,2,1)
#%% Load model
from keras.models import load_model
model_disp = load_model('DNN_model_Disp.h5')
model_velo = load_model('DNN_model_Velo.h5')
model_acce = load_model('DNN_model_Acce.h5')

#%% Prediction
y_pred_disp  = np.exp(model_disp.predict([Hysteretic_info_total, 
                                          X1_P, X1_MR, X1_Sa])) # displacement, m
y_pred_velo  = np.exp(model_velo.predict([Hysteretic_info_total, 
                                          X1_P, X1_MR, X1_Sa])) # Velocity, m/s
y_pred_acce  = np.exp(model_acce.predict([Hysteretic_info_total,
                                          X1_P, X1_MR, X1_Sa])) # Acceleration m/s2