"""
Taeyong Kim by Seoul National University

This script is for generate the DNN model for estimating structural responses
under ground motion.
"""


import numpy as np
import sqlite3
import random
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

NumberofGM = 2
del input_db
#%% Fetch output data (i.e. displacement from DB)

RSN = np.random.choice(RSN_training, NumberofGM, replace=False)
RSN = [int(x) for x in RSN]
RSN.sort()

RSN2 = ', '.join(str(x) for x in RSN)

Query_String = "select Demand_ID, AC.SDOF_R_ID, AC.\"Record Sequence Number\", Seismic_Demands.D "\
               "FROM (select * from Analysis_Cases WHERE Damping_ID=4 and SDOF_R_ID<=27090 and GM_Channel=1 and\"Record Sequence Number\" in (" + RSN2 + ")) as AC"\
               "    JOIN Seismic_Demands "\
               "        ON Seismic_Demands.rowid = AC.Demand_ID "\
               "    order by AC.\"Record Sequence Number\""\
               
db1.execute(Query_String) 
input_db  = db1.fetchall()
input_db = np.asarray(input_db)
Dis_transient_total = input_db[:,3]
Y2_index_1 =np.log(Dis_transient_total)

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

#%% ANN will be construct for each result
# Importing the Keras libraries and packages

from keras.layers import Input
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Activation, Dense
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
def relu(x):
    return Activation('relu')(x)

def linear(x):
    return Activation('linear')(x)

def tanh(x):
    return Activation('tanh')(x)
# Initialising the CNN for hysteretic model

hysteretic_input = Input(shape = (80, 2, 1))
hysteretic_prev = hysteretic_input

hysteretic1 = Convolution2D(32, (2, 2),padding = 'SAME')(hysteretic_prev)

hysteretic2 = Convolution2D(16, (4, 2),padding = 'SAME')(hysteretic_prev)

hysteretic3 = Convolution2D(8, (8, 2),padding = 'SAME')(hysteretic_prev)

hysteretic4 = Convolution2D(4, (16, 2),padding = 'SAME')(hysteretic_prev)

hysteretic_prev2 = concatenate([hysteretic_prev,hysteretic1,hysteretic2,hysteretic3,hysteretic4],3)
hysteretic_prev2 = tanh(hysteretic_prev2)

hysteretic1 = Convolution2D(32, (2, 2),padding = 'SAME')(hysteretic_prev2)

hysteretic2 = Convolution2D(16, (4, 2),padding = 'SAME')(hysteretic_prev2)

hysteretic3 = Convolution2D(8, (8, 2),padding = 'SAME')(hysteretic_prev2)

hysteretic4 = Convolution2D(4, (16, 2),padding = 'SAME')(hysteretic_prev2)

hysteretic_prev3 = concatenate([hysteretic_prev2,hysteretic1,hysteretic2,hysteretic3,hysteretic4],3)
hysteretic_prev3 = relu(hysteretic_prev3)
hysteretic_prev3 = BatchNormalization(axis=-1)(hysteretic_prev3)

hysteretic1 = Convolution2D(4, (2, 2),padding = 'SAME')(hysteretic_prev3)
hysteretic1 = MaxPooling2D(pool_size=(2,1), padding='valid')(hysteretic1)
hysteretic1 = relu(hysteretic1)

hysteretic1 = BatchNormalization(axis=-1)(hysteretic1)
hysteretic1 = Convolution2D(8, (2, 2),padding = 'SAME')(hysteretic1)
hysteretic1 = MaxPooling2D(pool_size=(2,1), padding='valid')(hysteretic1)
hysteretic1 = relu(hysteretic1)

hysteretic1 = BatchNormalization(axis=-1)(hysteretic1)
hysteretic1 = Convolution2D(16, (2, 2),padding = 'SAME')(hysteretic1)
hysteretic1 = MaxPooling2D(pool_size=(2,1), padding='valid')(hysteretic1)
hysteretic1 = relu(hysteretic1)


hysteretic2 = Convolution2D(4, (4, 2),padding = 'SAME')(hysteretic_prev3)
hysteretic2 = MaxPooling2D(pool_size=(2,1), padding='valid')(hysteretic2)
hysteretic2 = relu(hysteretic2)

hysteretic2 = BatchNormalization(axis=-1)(hysteretic2)
hysteretic2 = Convolution2D(8, (4, 2),padding = 'SAME')(hysteretic2)
hysteretic2 = MaxPooling2D(pool_size=(2,1), padding='valid')(hysteretic2)
hysteretic2 = relu(hysteretic2)

hysteretic2 = BatchNormalization(axis=-1)(hysteretic2)
hysteretic2 = Convolution2D(16, (4, 2),padding = 'SAME')(hysteretic2)
hysteretic2 = MaxPooling2D(pool_size=(2,1), padding='valid')(hysteretic2)
hysteretic2 = relu(hysteretic2)

hysteretic3 = Convolution2D(4, (8, 2),padding = 'SAME')(hysteretic_prev3)
hysteretic3 = MaxPooling2D(pool_size=(2,1), padding='valid')(hysteretic3)
hysteretic3 = relu(hysteretic3)

hysteretic3 = BatchNormalization(axis=-1)(hysteretic3)
hysteretic3 = Convolution2D(8, (8, 2),padding = 'SAME')(hysteretic3)
hysteretic3 = MaxPooling2D(pool_size=(2,1), padding='valid')(hysteretic3)
hysteretic3 = relu(hysteretic3)

hysteretic3 = BatchNormalization(axis=-1)(hysteretic3)
hysteretic3 = Convolution2D(16, (8, 2),padding = 'SAME')(hysteretic3)
hysteretic3 = MaxPooling2D(pool_size=(2,1), padding='valid')(hysteretic3)
hysteretic3 = relu(hysteretic3)


hysteretic4 = Convolution2D(4, (16, 2),padding = 'SAME')(hysteretic_prev3)
hysteretic4 = MaxPooling2D(pool_size=(2,1), padding='valid')(hysteretic4)
hysteretic4 = relu(hysteretic4)

hysteretic4 = BatchNormalization(axis=-1)(hysteretic4)
hysteretic4 = Convolution2D(8, (16, 2),padding = 'SAME')(hysteretic4)
hysteretic4 = MaxPooling2D(pool_size=(2,1), padding='valid')(hysteretic4)
hysteretic4 = relu(hysteretic4)

hysteretic4 = BatchNormalization(axis=-1)(hysteretic4)
hysteretic4 = Convolution2D(16, (16, 2),padding = 'SAME')(hysteretic4)
hysteretic4 = MaxPooling2D(pool_size=(2,1), padding='valid')(hysteretic4)
hysteretic4 = relu(hysteretic4)

hysteretic1 = Flatten()(hysteretic1)
hysteretic2 = Flatten()(hysteretic2) 
hysteretic3 = Flatten()(hysteretic3) 
hysteretic4 = Flatten()(hysteretic4) 

hysteretic1 = BatchNormalization(axis=-1)(hysteretic1)
hysteretic1 = Dense(units = 64)(hysteretic1)
hysteretic1 = relu(hysteretic1)

hysteretic1 = BatchNormalization(axis=-1)(hysteretic1)
hysteretic1 = Dense(units = 16)(hysteretic1)
hysteretic1 = relu(hysteretic1)

hysteretic2 = BatchNormalization(axis=-1)(hysteretic2)
hysteretic2 = Dense(units = 64)(hysteretic2)
hysteretic2 = relu(hysteretic2)

hysteretic2 = BatchNormalization(axis=-1)(hysteretic2)
hysteretic2 = Dense(units = 16)(hysteretic2)
hysteretic2 = relu(hysteretic2)

hysteretic3 = BatchNormalization(axis=-1)(hysteretic3)
hysteretic3 = Dense(units = 64)(hysteretic3)
hysteretic3 = relu(hysteretic3)

hysteretic3 = BatchNormalization(axis=-1)(hysteretic3)
hysteretic3 = Dense(units = 16)(hysteretic3)
hysteretic3 = relu(hysteretic3)

hysteretic4 = BatchNormalization(axis=-1)(hysteretic4)
hysteretic4 = Dense(units = 64)(hysteretic4)
hysteretic4 = relu(hysteretic4)

hysteretic4 = BatchNormalization(axis=-1)(hysteretic4)
hysteretic4 = Dense(units = 16)(hysteretic4)
hysteretic4 = relu(hysteretic4)

hysteretic_prev4 = concatenate([hysteretic1,hysteretic2,hysteretic3,hysteretic4])

hysteretic = BatchNormalization(axis=-1)(hysteretic_prev4)
hysteretic = Dense(units = 48)(hysteretic)
hysteretic = relu(hysteretic)

hysteretic = BatchNormalization(axis=-1)(hysteretic)
hysteretic = Dense(units = 64)(hysteretic)
hysteretic = relu(hysteretic)


sa_input = Input(shape = (110,))
gm_sa = sa_input

gm_sa = BatchNormalization(axis=-1)(gm_sa)
gm_sa = Dense(units = 128)(gm_sa)
gm_sa = relu(gm_sa)

gm_sa = BatchNormalization(axis=-1)(gm_sa)
gm_sa = Dense(units = 256)(gm_sa)
gm_sa = relu(gm_sa)

gm_sa = BatchNormalization(axis=-1)(gm_sa)
gm_sa = Dense(units = 64)(gm_sa)
gm_sa = relu(gm_sa)


merged2 = concatenate([hysteretic, gm_sa])
merged2 = BatchNormalization(axis=-1)(merged2)

gm_sa_sa = Dense(units = 128)(merged2)
gm_sa_sa = relu(gm_sa_sa)

gm_sa_sa = BatchNormalization(axis=-1)(gm_sa_sa)
gm_sa_sa = Dense(units = 64)(gm_sa_sa)
gm_sa_sa = relu(gm_sa_sa)

gm_sa_sa = BatchNormalization(axis=-1)(gm_sa_sa)
gm_sa_sa = Dense(units = 32)(gm_sa_sa)
gm_sa_sa = relu(gm_sa_sa)


MR_input = Input(shape = (7,))
gm_mr = MR_input

gm_mr = BatchNormalization(axis=-1)(gm_mr)
gm_mr = Dense(units = 16)(gm_mr)
gm_mr = relu(gm_mr)

gm_mr = BatchNormalization(axis=-1)(gm_mr)
gm_mr = Dense(units = 32)(gm_mr)
gm_mr = relu(gm_mr)

gm_mr = BatchNormalization(axis=-1)(gm_mr)
gm_mr = Dense(units = 16)(gm_mr)
gm_mr = relu(gm_mr)


merged3 = concatenate([hysteretic, gm_mr])
merged3 = BatchNormalization(axis=-1)(merged3)

gm_mr_mr = Dense(units = 128)(merged3)
gm_mr_mr = relu(gm_mr_mr)

gm_mr_mr = BatchNormalization(axis=-1)(gm_mr_mr)
gm_mr_mr = Dense(units = 64)(gm_mr_mr)
gm_mr_mr = relu(gm_mr_mr)

gm_mr_mr = BatchNormalization(axis=-1)(gm_mr_mr)
gm_mr_mr = Dense(units = 32)(gm_mr_mr)
gm_mr_mr = relu(gm_mr_mr)


# Add new ground motion info
groundmotion_input = Input(shape = (3,))
groundmotion = groundmotion_input

groundmotion = BatchNormalization(axis=-1)(groundmotion)
groundmotion = Dense(units = 16)(groundmotion)
groundmotion = relu(groundmotion)

groundmotion = BatchNormalization(axis=-1)(groundmotion)
groundmotion = Dense(units = 32)(groundmotion)
groundmotion = relu(groundmotion)

groundmotion = BatchNormalization(axis=-1)(groundmotion)
groundmotion = Dense(units = 16)(groundmotion)
groundmotion = relu(groundmotion)


merged4 = concatenate([hysteretic, groundmotion])
merged4 = BatchNormalization(axis=-1)(merged4)

gm_p_p = Dense(units = 128)(merged4)
gm_p_p = relu(gm_p_p)

gm_p_p = BatchNormalization(axis=-1)(gm_p_p)
gm_p_p = Dense(units = 64)(gm_p_p)
gm_p_p = relu(gm_p_p)

gm_p_p = BatchNormalization(axis=-1)(gm_p_p)
gm_p_p = Dense(units = 32)(gm_p_p)
gm_p_p = relu(gm_p_p)


# Make convolute model
merged = concatenate([hysteretic, gm_p_p, gm_sa_sa, gm_mr_mr])
merged = BatchNormalization(axis=-1)(merged)

# Final - Full connection for Transient (Y1)
regressor = Dense(units = 256)(merged)
regressor = relu(regressor)

regressor = BatchNormalization(axis=-1)(regressor)
regressor = Dense(units = 128)(regressor)
regressor = relu(regressor)

regressor = BatchNormalization(axis=-1)(regressor)
regressor = Dense(units = 64)(regressor)
regressor = relu(regressor)

regressor = BatchNormalization(axis=-1)(regressor)
regressor = Dense(units = 32)(regressor)
regressor = relu(regressor)

regressor = BatchNormalization(axis=-1)(regressor)
regressor = Dense(units = 1)(regressor)
regressor = linear(regressor)

# Compiling the CNN
model = Model(inputs = [hysteretic_input, groundmotion_input, MR_input, sa_input], outputs= regressor)

model.compile(loss='mean_squared_error', optimizer='Adam')

#%% Data augmentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        height_shift_range =0.2,
        vertical_flip=True,
	fill_mode='reflect')

#%% See the CNN results
batches=0
Hys_batch = []
X1_P_batch = []
X1_MR_batch = []
X1_Sa_batch = []
y_batch = []

index = np.arange(0,len(Y2_index_1))

for x_batch, index_batch in train_datagen.flow(Hysteretic_info_total, index, batch_size=3000):
    Hys_batch.append(x_batch)
    X1_P_batch.append(X1_P[index_batch,:])
    X1_MR_batch.append(X1_MR[index_batch,:])
    X1_Sa_batch.append(X1_Sa[index_batch,:])
    y_batch.append( Y2_index_1[index_batch,])
    batches += 1
    if batches >= len(index) / 3000:
        # we need to break the loop by hand because
        # the generator loops indefinitely
        break
del index_batch, x_batch, batches

Hys_batch = np.concatenate(np.asarray(Hys_batch)).astype(None)
X1_P_batch = np.concatenate(np.asarray(X1_P_batch)).astype(None)
X1_MR_batch = np.concatenate(np.asarray(X1_MR_batch)).astype(None)
X1_Sa_batch = np.concatenate(np.asarray(X1_Sa_batch)).astype(None)
y_batch = np.concatenate(np.asarray(y_batch)).astype(None)

history_callback = model.fit([Hys_batch, X1_P_batch, X1_MR_batch, X1_Sa_batch], y_batch, batch_size = 512, epochs = 10)

    
