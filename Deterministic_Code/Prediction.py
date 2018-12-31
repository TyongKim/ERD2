"""
Taeyong Kim by Seoul National University

This script is for generate the DNN model for estimating structural responses
under ground motion.
"""


import numpy as np
import sqlite3
import random
import matplotlib.pyplot as plt

conn = sqlite3.connect("SDDB_v1.02.db") # Import database from http:/ERD2.snu.ac.kr
db1 = conn.cursor()

#%% Find number of GM

Query_String = "select Demand_ID, EQ_Events.\"Earthquake Magnitude\", EQ_Stations.\"Preferred NEHRP Based on Vs30\", Analysis_Cases.\"Record Sequence Number\" "\
               "FROM Analysis_Cases "\
               "    JOIN Seismic_Demands "\
               "        ON Seismic_Demands.rowid = Analysis_Cases.Demand_ID "\
               "    JOIN SDOF_Realization "\
               "        ON SDOF_Realization.SDOF_R_ID = Analysis_Cases.SDOF_R_ID "\
               "    JOIN EQ_Records "\
               "        ON EQ_Records.\"Record Sequence Number\" = Analysis_Cases.\"Record Sequence Number\" "\
               "    JOIN EQ_Events "\
               "        ON EQ_Events.EQID = EQ_Records.EQID "\
               "    JOIN EQ_Stations "\
               "        ON EQ_Records.\"Station Sequence Number\" = EQ_Stations.\"Station Sequence Number\" "\
               "WHERE Analysis_Cases.Damping_ID=4 and Analysis_Cases.SDOF_R_ID=1 and GM_Channel=1"\
               "    order by Analysis_Cases.\"Record Sequence Number\""\

db1.execute(Query_String) 
input_db  = db1.fetchall()
input_db = np.asarray(input_db)

RSN = input_db[:,3]
RSN = [str(i) for i in RSN]
RSN = np.asarray(RSN)

# Randomly select the dataset
random.shuffle(RSN)

RSN_training = RSN[0:np.int(len(RSN)*0.8)]

NumberofGM = 1
del input_db
#%% Fetch output data (i.e. displacement from DB)

RSN = np.random.choice(RSN_training, NumberofGM, replace=False)
RSN = [int(x) for x in RSN]
RSN.sort()

RSN2 = ', '.join(str(x) for x in RSN)

Query_String = "select Demand_ID, AC.SDOF_R_ID, AC.\"Record Sequence Number\", Seismic_Demands.D, Seismic_Demands.V , Seismic_Demands.A "\
               "FROM (select * from Analysis_Cases WHERE Damping_ID=4 and SDOF_R_ID<=54090 and GM_Channel=1 and\"Record Sequence Number\" in (" + RSN2 + ")) as AC"\
               "    JOIN Seismic_Demands "\
               "        ON Seismic_Demands.rowid = AC.Demand_ID "\
               "    order by AC.\"Record Sequence Number\""\
               
db1.execute(Query_String) 
input_db  = db1.fetchall()
input_db = np.asarray(input_db)
Dis_transient_total = input_db[:,3]
Vel_transient_total = input_db[:,4]
Acc_transient_total = input_db[:,5]

Y2_index_1 =np.log(Dis_transient_total)
Y2_index_2 =np.log(Vel_transient_total)
Y2_index_3 =np.log(Acc_transient_total)
del Dis_transient_total, Vel_transient_total, Acc_transient_total
#%% Hysteretic Behaviour
Hysteretic_ElemForc = np.load('./Hysteresis-Force.npy')
Hysteretic_NodeDisp = np.load('./Hysteresis-Disp.npy')

Hysteretic = []
for i in range(np.int(np.size(Hysteretic_ElemForc)/len(Hysteretic_ElemForc))):
    tmp = np.transpose(np.asarray([Hysteretic_NodeDisp[:,i], Hysteretic_ElemForc[:,i]]))
    Hysteretic.append(tmp)
    
Hysteretic = np.asarray(Hysteretic)

Hysteretic_info_total=[]
for ii in range(NumberofGM):
    for jj in range(len(Hysteretic)):
        
        Hysteretic_info_total.append(Hysteretic[jj,:,:])

Hysteretic_info_total = np.asarray(Hysteretic_info_total)   
Hysteretic_info_total = Hysteretic_info_total.reshape(int(len(Hysteretic)*NumberofGM),80,2,1)

del tmp, i, Hysteretic_ElemForc, Hysteretic_NodeDisp, Hysteretic, ii, jj

#%% Fetch Ground motion information

Query_String = "select Demand_ID, EQ_Events.\"Earthquake Magnitude\", EQ_Stations.\"Preferred NEHRP Based on Vs30\", Analysis_Cases.\"Record Sequence Number\" "\
               "FROM Analysis_Cases "\
               "    JOIN Seismic_Demands "\
               "        ON Seismic_Demands.rowid = Analysis_Cases.Demand_ID "\
               "    JOIN SDOF_Realization "\
               "        ON SDOF_Realization.SDOF_R_ID = Analysis_Cases.SDOF_R_ID "\
               "    JOIN EQ_Records "\
               "        ON EQ_Records.\"Record Sequence Number\" = Analysis_Cases.\"Record Sequence Number\" "\
               "    JOIN EQ_Events "\
               "        ON EQ_Events.EQID = EQ_Records.EQID "\
               "    JOIN EQ_Stations "\
               "        ON EQ_Records.\"Station Sequence Number\" = EQ_Stations.\"Station Sequence Number\" "\
               "WHERE Analysis_Cases.Damping_ID=4 and Analysis_Cases.SDOF_R_ID=1 and Analysis_Cases.\"Record Sequence Number\" in (" + RSN2 + ") and GM_Channel=1"\
               "    order by Analysis_Cases.\"Record Sequence Number\""\

db1.execute(Query_String) 
input_db  = db1.fetchall()
input_db = np.asarray(input_db)

Magnitude = input_db[:,1]
Magnitude = [float(i) for i in Magnitude]

Site_info_total = input_db[:,2]
Site_info_total = [str(i) for i in Site_info_total]
Site_info_total = np.asarray(Site_info_total)


Site= []
for ii in range(len(Site_info_total)):
    temp_site = Site_info_total[ii]
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
del ii, temp_site, Site_info_total
Site = np.asarray(Site)

targets = np.array(Site).reshape(-1)    
Site = np.eye(5)[targets]

del targets 

Ground_info_total_b_temp = []
for ii in range(len(RSN)):
    temp_RSN = np.int(RSN[ii])
    Query_String = "SELECT * FROM EQ_Records WHERE \"Record Sequence Number\" = %d " %(temp_RSN)
    db1.execute(Query_String) 
    temp_value = db1.fetchall()
    Ground_info = np.r_[Magnitude[ii], temp_value[0][3], Site[ii,:], temp_value[0][45:158]]    # Distance +PGA PGV PGD + Spectral acceleration
    Ground_info_total_b_temp.append(Ground_info)
    
Ground_info_total_b_temp = np.asarray(Ground_info_total_b_temp)

Ground_info_total_b=[]
for i in range(len(RSN)):
    for j in range(int(len(Hysteretic_info_total)/len(RSN))):
        Ground_info_total_b.append(Ground_info_total_b_temp[i,:])
Ground_info_total_b = np.asarray(Ground_info_total_b)


X1_P  = np.log(Ground_info_total_b[:,7:10])
X1_MR = Ground_info_total_b[:,0:7]
X1_Sa = np.log(Ground_info_total_b[:,10:])

del Ground_info_total_b, Ground_info_total_b_temp, i, j, ii, Ground_info, Query_String, Magnitude, Site, temp_value, temp_RSN

#%% Load model
from keras.models import load_model
model_disp = load_model('DNN_model_Disp.h5')
model_velo = load_model('DNN_model_Velo.h5')
model_acce = load_model('DNN_model_Acce.h5')

#%% Predict
y_pred_disp  = model_disp.predict([Hysteretic_info_total, X1_P, X1_MR, X1_Sa])
y_pred_velo  = model_velo.predict([Hysteretic_info_total, X1_P, X1_MR, X1_Sa])
y_pred_acce  = model_acce.predict([Hysteretic_info_total, X1_P, X1_MR, X1_Sa])

#%% Scatter plot
E_disp = []
E_velo = []
E_acce = []
for ii in range(len(y_pred_disp)):
    E_disp.append((Y2_index_1[ii] - y_pred_disp[ii]))
    E_velo.append((Y2_index_2[ii] - y_pred_velo[ii]))
    E_acce.append((Y2_index_3[ii] - y_pred_acce[ii]))
E_disp = np.asarray(E_disp)
E_velo = np.asarray(E_velo)
E_acce = np.asarray(E_acce)

x_plot = np.arange(0,len(y_pred_disp))
plt.close('all')
plt.figure()
plt.scatter(x_plot,E_disp)

plt.figure()
plt.scatter(x_plot,E_velo)

plt.figure()
plt.scatter(x_plot,E_acce)


