"""
This script is to provide how to generate hysteresis of the m-BWBN model in a 
form of an input of the ERD-net. The generated hystereses are saved in a database.

Written by Taeyong Kim at Ajou University

March 19, 2023

"""
# Import libraries
import numpy as np
import sqlite3

# Backward Euler solution scheme in the appendix of the reference.
from mBWBN_quasi_cyclic_python import mBWBN_quasi_cyclic_python

#%% Construct a database to store the generated hysteresis
# SQLite is used in this task.

g = 9.8; # gravitational acceleration

# Define database   
Name_db = 'Hysteretic_BWBN.db'; # Name of database for saving hysteresis
conn2 = sqlite3.connect(Name_db)
cur = conn2.cursor()

# Make database for hysteresis of structural system
SI_sql = """ CREATE TABLE Structural_info (ID integer PRIMARY KEY, 
                                          \"Period(s)\" real NOT NULL,
                                          \"Stiffness(g/m)\" real NOT NULL,
                                          \"Yield_force(g)\" real NOT NULL,
                                           Post_yield_stiffness real NOT NULL,
                                           deltaNu real NOT NULL,
                                           deltaEta real NOT NULL,
                                           zeta0 real NOT NULL,
                                           p real NOT NULL)"""


# SQL to insert the value to the database
SI_val_sql = "INSERT INTO Structural_info (ID, \"Period(s)\", \"Stiffness(g/m)\", \
                                            \"Yield_force(g)\", Post_yield_stiffness, \
                                            deltaNu, deltaEta, zeta0, p) \
                                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "

# We use 241 by 2 hysteretic behaviors 
tmp_Hy_sql_force = ''
tmp_Hy_sql_displ = ''
for ii in range(241):
    tmp_force = '\"Force(g)_%d\" real NOT NULL, ' %(ii)
    tmp_Hy_sql_force = (tmp_Hy_sql_force + tmp_force)
    tmp_displ = '\"Displ(m)_%d\" real NOT NULL, ' %(ii)
    tmp_Hy_sql_displ = (tmp_Hy_sql_displ + tmp_displ)
        
Hy_sql_for = (""" CREATE TABLE Hys_behavior_force (System_index integer NOT NULL,
                                                  Random_num integer NOT NULL, \n"""+tmp_Hy_sql_force +
                                                """FOREIGN KEY (System_index) REFERENCES Structural_info (ID)) """)
    
Hy_sql_dis = (""" CREATE TABLE Hys_behavior_displ (System_index integer NOT NULL,
                                                  Random_num integer NOT NULL, \n"""+tmp_Hy_sql_displ +
                                                """FOREIGN KEY (System_index) REFERENCES Structural_info (ID)) """)
    

tmp_hy_val_force = ''
tmp_hy_val_displ = ''
tmp_values = 'VALUES (?, ?, '
for ii in range(241):
    if ii==240:
        tmp_values = (tmp_values+'? )')
        tmp_force = ' \"Force(g)_%d\" ) ' %(ii)
        tmp_hy_val_force = (tmp_hy_val_force + tmp_force)
        tmp_displ = ' \"Displ(m)_%d\" ) ' %(ii)
        tmp_hy_val_displ = (tmp_hy_val_displ + tmp_displ)         
    else:
        tmp_values = (tmp_values+'?, ')
        tmp_force = ' \"Force(g)_%d\", ' %(ii)
        tmp_hy_val_force = (tmp_hy_val_force + tmp_force)
        tmp_displ = ' \"Displ(m)_%d\", ' %(ii)
        tmp_hy_val_displ = (tmp_hy_val_displ + tmp_displ)        

hy_val_force_sql = "INSERT INTO Hys_behavior_force (System_index, Random_num," +tmp_hy_val_force + tmp_values
hy_val_displ_sql = "INSERT INTO Hys_behavior_displ (System_index, Random_num," +tmp_hy_val_displ + tmp_values

cur.execute(SI_sql)
cur.execute(Hy_sql_for)
cur.execute(Hy_sql_dis)
conn2.commit()                

#%% Input structural parameters for m-BWBN model
# Hysteretic parameter of interest (m-BWBN model, sitff/strength degradations + Pinching)
# Total of hysteretic system 40*10*4*81 = 129,600

# Period
str_period = np.array([0.05, 0.055, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.20, 
                       0.23, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46, 0.5, 0.55, 0.60, 
                       0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0, 1.2, 1.5, 2.0, 
                       2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]); # Total of 40

# Normalized stiffness (unit g, Normalized yield force (g)/ displzcement (m))
str_stiffness = 1/g/(str_period/(2*np.pi))**2; # Normalized stiffness

# Normalized yield force (unit g)
str_yield_force = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75,
                            1.0, 1.25, 1.5]); #% 10

# Post yield stiffness ratio
str_post_yield = [0.0, 0.05, 0.2, 0.5];  # 4

# m-BWBN parameters
str_n = 3; # fix
str_del_Nu = np.array([0.0, 0.1, 0.36])#; % 3
str_del_Eta = np.array([0.0, 0.1, 0.39])#; % 3
str_pinch_q = 0.027#; % fix
str_pinch_shi = 0.117#; % fix
str_pinch_delshi = 0.003#; % fix
str_pinch = np.array([[0, 0, 0.025],
             [0.58, 0.33, 0.012],
             [0.62, 0.22, 0.45],
             [0.65, 1.20, 0.015],
             [0.70, 1.38, 0.38],
             [0.80, 0.24, 0.019],
             [0.92, 0.34, 0.6],
             [0.87, 1.30, 0.025],
             [0.91, 1.36, 0.7]])#; % zeta0(0.7, 0.9), p (0.3,1.38), lambda (0.015, 0.4)


#%% Make combination of structural systems (40*10*4*81 = 129,600)
# Save structural information
idx = 0; # index of a structural system
struc_total = []; param_total = []
for ii in range(len(str_period)): # period
    for jj in range(len(str_yield_force)):# yield force
        for kk in range(len(str_post_yield)): # post yield stiffness ratio
            for pp in range(len(str_del_Nu)): # del_nu strnegth degradation
                for qq in range(len(str_del_Eta)): # del_eta stiffness degradation
                    for rr in range(len(str_pinch)): # Pinching effect

                        tmp_period = str_period[ii]#; % period
                        tmp_yield_force = str_yield_force[jj]#; % Normalized yield force
                        tmp_post_yield = str_post_yield[kk]#; % post yield stiffness ratio
                        tmp_del_Nu = str_del_Nu[pp]#; %del nu
                        tmp_del_Eta = str_del_Eta[qq]#; % del Eta
                        tmp_pinch_zeta = str_pinch[rr,0]#; % pinch_zeta
                        tmp_pinch_p = str_pinch[rr,1]#; % pinch p
                        tmp_pinch_lambda = str_pinch[rr,2]#; % pinch lambda

                        # parameters for structural system
                        tmp_parameters = np.array([tmp_post_yield,    # % Post-yield stiffnes ratio
                                                   tmp_yield_force*g,# % Yield force
                                                   tmp_period,       # % Period
                                                   0.5,              # % Beta (fix)
                                                   0.5,              # % Gamma (fix)
                                                   3,                # % n (fix)
                                                   tmp_del_Nu,       # % delta Nu
                                                   tmp_del_Eta,      # % delta Eta
                                                   tmp_pinch_zeta,   # % Xi
                                                   tmp_pinch_p,      # % p
                                                   0.027,            # % q (fix)
                                                   0.117,            # % Shi (fix)
                                                   0.003,            # % delta Shi (fix)
                                                   tmp_pinch_lambda])#; % Lambda
                        
                        param_total.append(tmp_parameters)

                        # stiffness derived from the period -->
                        # normalized pushover displacement
                        k0 = (1/g/(tmp_parameters[2]/(2*np.pi))**2)*g;
                        
                        # Save the structural system variables to the database
                        tmp_struc = np.r_[[int(idx), tmp_period, k0/g, tmp_yield_force,
                                            tmp_post_yield, tmp_del_Nu,
                                            tmp_del_Eta, tmp_pinch_zeta, tmp_pinch_p]]
                        struc_total.append(tmp_struc)

                        
                        idx = idx+1 #% index of a structural system
                                     
cur.executemany(SI_val_sql,struc_total)
conn2.commit()
param_total = np.asarray(param_total)
struc_total = np.asarray(struc_total)
#%% Define function for a quasi static cyclic analysis

def loop_worker(jj, rsn): # jj: index of structural system; rsn: random sequence
    
    # Parameters of a structural system
    tmp_parameters = param_total[jj,:]
    tmp_struc = struc_total[jj,:]
    # normalized pushover displacement
    k0 = (1/g/(tmp_parameters[2]/(2*np.pi))**2)*g;
    
    # Displacement steps for push curve
    Displ_pushover = np.arange(0,5+0.0001,0.0001)
    Displ_pushover = Displ_pushover*tmp_parameters[1]/k0; # Normalize the input displacement value (m)
    
    # Peform pushover analysis
    Force_pushover = mBWBN_quasi_cyclic_python(tmp_parameters, Displ_pushover); # m/s^2
    
    # Find the intersection point between pushover
    # analysis and the 90% of initial stiffness
    Init_stiff = ((Force_pushover[1]-Force_pushover[0]) 
                 /(Displ_pushover[1]-Displ_pushover[0]));
    
    # 90% of initial stiffness
    Init_stiff_90 = Init_stiff*0.9; # 90% of initial stiffness
    
    # Find the intersection point between auxillary line and pushover curve
    tmp_displ = np.arange(0,Displ_pushover[-1],Displ_pushover[-1]/10000)
    inter_force = np.interp(tmp_displ, Displ_pushover, Force_pushover); # interpolate
    for_intersect = np.abs(tmp_displ*Init_stiff_90 - inter_force);
    intersection_point = np.where(for_intersect<1e-2); #% index of intersection point
    intersection_point_final = intersection_point[-1][-1];
    
    optimal_yield_displ = tmp_displ[intersection_point_final]; # Yield displcement                            
    
    D_refer = optimal_yield_displ# Reference point (m)
    # Put randomness in yield displacement
    D_refer = np.random.uniform(0.98,1.1)*D_refer
    
    TMP_DISP = []
    tmp_disp = np.arange(0,0.51,0.25)
    TMP_DISP = np.r_[TMP_DISP, tmp_disp, np.flip(tmp_disp[0:-1]), -tmp_disp[1:],-np.flip(tmp_disp[1:-1])]
    TMP_DISP = np.r_[TMP_DISP, tmp_disp, np.flip(tmp_disp[0:-1]), -tmp_disp[1:],-np.flip(tmp_disp[1:-1])]
    tmp_disp = np.arange(0,1.01,0.25)
    TMP_DISP = np.r_[TMP_DISP, tmp_disp, np.flip(tmp_disp[0:-1]), -tmp_disp[1:],-np.flip(tmp_disp[1:-1])]
    TMP_DISP = np.r_[TMP_DISP, tmp_disp, np.flip(tmp_disp[0:-1]), -tmp_disp[1:],-np.flip(tmp_disp[1:-1])]
    tmp_disp = np.arange(0,1.51,0.25)
    TMP_DISP = np.r_[TMP_DISP, tmp_disp, np.flip(tmp_disp[0:-1]), -tmp_disp[1:],-np.flip(tmp_disp[1:-1])]
    TMP_DISP = np.r_[TMP_DISP, tmp_disp, np.flip(tmp_disp[0:-1]), -tmp_disp[1:],-np.flip(tmp_disp[1:-1])]
    tmp_disp = np.arange(0,2.01,0.25)
    TMP_DISP = np.r_[TMP_DISP, tmp_disp, np.flip(tmp_disp[0:-1]), -tmp_disp[1:],-np.flip(tmp_disp[1:-1])]
    TMP_DISP = np.r_[TMP_DISP, tmp_disp, np.flip(tmp_disp[0:-1]), -tmp_disp[1:],-np.flip(tmp_disp[1:-1])]
    tmp_disp = np.arange(0,2.51,0.25)
    TMP_DISP = np.r_[TMP_DISP, tmp_disp, np.flip(tmp_disp[0:-1]), -tmp_disp[1:],-np.flip(tmp_disp[1:-1])]
    TMP_DISP = np.r_[TMP_DISP, tmp_disp, np.flip(tmp_disp[0:-1]), -tmp_disp[1:],-np.flip(tmp_disp[:-1])]
    
    Displ_step = TMP_DISP*D_refer                            
    
    # Give some randomness
    Force_step = np.zeros([241,1]); Force_step[240,0] = np.nan; ttp_max_force = 0
    while (np.sum(np.isnan(Force_step))>=1) or (np.max(np.abs(Force_step))>ttp_max_force):
        Displ_step_random = [];
        for tt in range(len(Displ_step)):
            tmp_displ_step = Displ_step[tt]
            if tt==0:
                tmp_displ_step_rand = 0; #% normal random, c.o.v 3%
            elif tmp_displ_step==0:
                tmp_displ_step_rand = np.random.normal(0,np.abs(Displ_step[1])*0.005); #% normal random, c.o.v 3%
            else:
                tmp_displ_step_rand = np.random.normal(tmp_displ_step,np.abs(tmp_displ_step)*0.005); #% normal random, c.o.v 3%
                                                    
            Displ_step_random.append(tmp_displ_step_rand) #; % unit: m
        
        Displ_step_random = np.asarray(Displ_step_random)
    
        # Perfrom quasi_cyclic analysis for DNN input
        # The values should be saved in the unit of (m, g) for displacement and
        # normalized force, respectively
        Force_step = mBWBN_quasi_cyclic_python(tmp_parameters, Displ_step_random); #% m/s^2
        
        # This is to consider crazy hysteresis
        ttp_yield_force = tmp_parameters[1] #;
        ttp_yield_displ = ttp_yield_force/k0;
        ttp_max_force = ttp_yield_force + k0*tmp_parameters[0]*(np.max(np.abs(Displ_step_random))-ttp_yield_displ);
    
    Force_step = np.asarray(Force_step)
    Normalized_Force_step = Force_step/g; # % Normalized the esitmated force (g)
    
    # Save the results to one array
    # Save the structural system variables to the database
    idx = tmp_struc[0]
    Displ_step_value = np.r_[int(idx), int(rsn+1), Displ_step_random]    
    Force_step_value = np.r_[int(idx), int(rsn+1), Normalized_Force_step]
        
    Results = [Displ_step_value, Force_step_value]
    return Results


#%% Run analysis and save the results
result_list_force = []; result_list_displ = []
for ii in range(len(struc_total)): # structual systmes
    for jj in range(2): # Randomness
        result_list = loop_worker(ii, jj)
        result_list_force.append(result_list[1])
        result_list_displ.append(result_list[0])            
                


cur.executemany(hy_val_force_sql,result_list_force) 
cur.executemany(hy_val_displ_sql,result_list_displ)     

conn2.commit()                                                                        
conn2.close() # close the database


