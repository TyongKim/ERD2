"""
Input script to predict seismic responses of a structural system.
Seismic responses: Transient maximum displacement, velocity, acceleration.

Define the corresponding structural system and ground motions.
After running this script, 'Code_Main_run.py' code needs to be performed.

Developed by Taeyong Kim from the Seoul National Universtiy
chs5566@snu.ac.kr
June 15, 2020
"""

# Basic libraries
import numpy as np
import pandas as pd
#%% Structural system
"""
###############################################################################
# Define a structural system                                                  #
# Hysteretic model                                                            #
# Period (s)                                                                  #
# Yield force (g)                                                             #
# post yield stiffness ratio                                                  #
###############################################################################

###############################################################################
Please select one of the hysteretic model among the following three models
HM1: Linear
HM2: Bilinear
HM3: Bilinear with stiffness degradation

Type the properties of structural system
If you select HM1, only one parameter is requried, otherwise three parameters 
are needed. Please note that when the values that are not listed in the list 
are typed, the value is automatically changed to the adjacent one in terms of 
the Euclidian distance. The values are selected from the following list
Period list
[ 0.05   0.055  0.06   0.065  0.067  0.07   0.075  0.08   0.085  0.09
  0.095  0.1    0.11   0.12   0.13   0.133  0.14   0.15   0.16   0.17
  0.18   0.19   0.2    0.22   0.24   0.25   0.26   0.28   0.29   0.3
  0.32   0.34   0.35   0.36   0.38   0.4    0.42   0.44   0.45   0.46
  0.48   0.5    0.55   0.6    0.65   0.667  0.7    0.75   0.8    0.85
  0.9    0.95   1.     1.1    1.2    1.3    1.4    1.5    1.6    1.7
  1.8    1.9    2.     2.2    2.4    2.5    2.6    2.8    3.     3.2
  3.4    3.5    3.6    3.8    4.     4.2    4.4    4.6    4.8    5.
  5.5    6.     6.5    7.     7.5    8.     8.5    9.     9.5   10.   ]

Yield force list
[0.05 0.1  0.15 0.2  0.25 0.3  0.35 0.4  0.45 0.5  0.55 0.6  0.65 0.7
 0.75 0.8  0.85 0.9  0.95 1.   1.05 1.1  1.15 1.2  1.25 1.3  1.35 1.4
 1.45 1.5 ]

Post yield stiffness ratio list
[0.   0.02 0.05 0.1  0.15 0.2  0.25 0.3  0.4  0.5 ]
###############################################################################
"""
Hysteretic_model = 'HM2' # three different models
Hyst_period = 0.2 
Hyst_yield_force = 0.1
Hyst_post_yield = 0.05

#%% Seismic hazard information
"""
###############################################################################
# Define seismic hazard                                                       #
# You may need three different types of seismic information                   #
# 1. PGA (g), PGV (cm/s), PGD (cm)                                            #
# 2. Magnitude, Epicenter distance (km), Soil type                            #
# 3. Response sepctrum (should contain 0.005 ~ 10 sec)                        #
# Please note that response spectrum file should match with the example file  #
# shown below                                                                 #
###############################################################################
"""

# 1970_Wrightwood_6074_Park_Dr_North ground motions
Input_GM_peak_info = [0.1445381, 8.512941, 1.251468] # PGA, PGV, and PGD
Input_earthquake_info = [5.33,	12.14, 'C'] # M, R, and Soil class
Input_RS_value = pd.read_csv('EX_RS_1970_Wrightwood.csv') # Spectral acceleartion