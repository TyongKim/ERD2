"""
This script let you generate many MDOF bridge systems

Written by Taeyong Kim at Ajou University
taeyongkim@ajou.ac.kr

"""
# Import libraries
import numpy as np

#%% Basic information
# Upper and lower limits of the structural characteristics
strcutre_limit = np.load('structure_limit.npy') # Table 1 of the reference
strcutre_limit = np.log(strcutre_limit) # Put log to address the skewness issue
damping = np.array([0.005, 0.01, 0.02, 0.03, 0.05]) # damping value

#%% Generate 
# This code randomly generates 5 bridges for each span
p3_value = []
for ii in range(5):

    mass_column = float(np.exp(np.random.uniform(strcutre_limit[0,0], strcutre_limit[0,1],1)))
    sitff_y_super = float(np.exp(np.random.uniform(strcutre_limit[1,0], strcutre_limit[1,1],1)))
    stiff_x_column = float(np.exp(np.random.uniform(strcutre_limit[2,0], strcutre_limit[2,1],1)))
    stiff_y_column = float(np.exp(np.random.uniform(strcutre_limit[3,0], strcutre_limit[3,1],1)))
    
    random_values = np.random.uniform(0.7,1.3, 11) # To put randomness when generating structure
    
    tmp_damping = damping[int(np.random.randint(5 ))]

    p3_value.append([mass_column, sitff_y_super, stiff_x_column, stiff_y_column, tmp_damping, random_values])


p4_value = []
for ii in range(5):

    mass_column = float(np.exp(np.random.uniform(strcutre_limit[0,0], strcutre_limit[0,1],1)))
    sitff_y_super = float(np.exp(np.random.uniform(strcutre_limit[1,0], strcutre_limit[1,1],1)))
    stiff_x_column = float(np.exp(np.random.uniform(strcutre_limit[2,0], strcutre_limit[2,1],1)))
    stiff_y_column = float(np.exp(np.random.uniform(strcutre_limit[3,0], strcutre_limit[3,1],1)))
    
    random_values = np.random.uniform(0.7,1.3, 11)
    
    tmp_damping = damping[int(np.random.randint(5 ))]
             
    p4_value.append([mass_column, sitff_y_super, stiff_x_column, stiff_y_column, tmp_damping, random_values])


p5_value = []
for ii in range(5):

    mass_column = float(np.exp(np.random.uniform(strcutre_limit[0,0], strcutre_limit[0,1],1)))
    sitff_y_super = float(np.exp(np.random.uniform(strcutre_limit[1,0], strcutre_limit[1,1],1)))
    stiff_x_column = float(np.exp(np.random.uniform(strcutre_limit[2,0], strcutre_limit[2,1],1)))
    stiff_y_column = float(np.exp(np.random.uniform(strcutre_limit[3,0], strcutre_limit[3,1],1)))
    
    random_values = np.random.uniform(0.7,1.3, 11)
    
    tmp_damping = damping[int(np.random.randint(5 ))]
        
    p5_value.append([mass_column, sitff_y_super, stiff_x_column, stiff_y_column, tmp_damping, random_values])


p6_value = []
for ii in range(5):

    mass_column = float(np.exp(np.random.uniform(strcutre_limit[0,0], strcutre_limit[0,1],1)))
    sitff_y_super = float(np.exp(np.random.uniform(strcutre_limit[1,0], strcutre_limit[1,1],1)))
    stiff_x_column = float(np.exp(np.random.uniform(strcutre_limit[2,0], strcutre_limit[2,1],1)))
    stiff_y_column = float(np.exp(np.random.uniform(strcutre_limit[3,0], strcutre_limit[3,1],1)))
    
    random_values = np.random.uniform(0.7,1.3, 11)
    
    tmp_damping = damping[int(np.random.randint(5 ))]

    p6_value.append([mass_column, sitff_y_super, stiff_x_column, stiff_y_column, tmp_damping, random_values])


p3_value = np.asarray(p3_value, dtype='object')
p4_value = np.asarray(p4_value, dtype='object')
p5_value = np.asarray(p5_value, dtype='object')
p6_value = np.asarray(p6_value, dtype='object')

# Save the MDOF systems
np.save('./generated_MDOF_systems/3_span.npy', p3_value)
np.save('./generated_MDOF_systems/4_span.npy', p4_value)
np.save('./generated_MDOF_systems/5_span.npy', p5_value)
np.save('./generated_MDOF_systems/6_span.npy', p6_value)

