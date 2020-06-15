"""

This script is to develop a DNN model to predict structural responses under an
earthquake. In order to perform this code, the "Seismic Demand Database (SDDB)" 
is required. The database can be downloaded at http://ERD2.snu.ac.kr

Moreover, although this script is written to develop a DNN model for the peak 
transient displacement of a structural system, other structural responses
such as acceleration and velocity are also employed as an output of a DNN 
model.

The deep learning model is developed based on the TensorFlow and Keras. Please
download the libraries.

The code is developed by Taeyong Kim from the Seoul National University
chs5566@snu.ac.kr

"""

# Basic libraries
import numpy as np
import sqlite3
import random

# If using MAC, this might be helpful
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#  In a local computer: set very small number, and in super computer use the entire training ground motions
NumberofGM = 2 # In super computer, 1199
Num_epochs = 10 # Number of epochs
#%%
###############################################################################
# Preprocessing
# 1. Select ground motions for training
# 2. Fetch Output dataset (i.e. seismic demands)
# 3. Fetch Input dataset (i.e. hysteretic behaviors, ground motion information)
###############################################################################

#%%
# Preprocessing: 1. Select ground motions for training
# Import database which is able to be downloaded at http:/ERD2.snu.ac.kr
conn = sqlite3.connect("/Users/taeyongkim/Desktop/SDDB_v1.02.db")
db1 = conn.cursor()

# Find number of GM using the predefined conditions
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

RSN = input_db[:,3] # Record sequence number (Total of 1,499 ground motions)
RSN = [str(i) for i in RSN]
RSN = np.asarray(RSN)

# Randomly select the dataset: 80% of the dataset is used for training
random.shuffle(RSN)
RSN_training = RSN[0:np.int(len(RSN)*0.8)]

#%%
# Preprocessing: 2. Fetch output data (i.e. displacement from DB)
RSN = np.random.choice(RSN_training, NumberofGM, replace=False)
RSN = [int(x) for x in RSN]
RSN.sort()

RSN2 = ', '.join(str(x) for x in RSN)

Query_String = "select Demand_ID, AC.SDOF_R_ID, AC.\"Record Sequence Number\", Seismic_Demands.D "\
               "FROM (select * from Analysis_Cases WHERE Damping_ID=4 and SDOF_R_ID<=54090 and GM_Channel=1 and\"Record Sequence Number\" in (" + RSN2 + ")) as AC"\
               "    JOIN Seismic_Demands "\
               "        ON Seismic_Demands.rowid = AC.Demand_ID "\
               "    order by AC.\"Record Sequence Number\""\
               
db1.execute(Query_String) 
input_db  = db1.fetchall()
input_db = np.asarray(input_db)
Dis_transient_total = input_db[:,3]
Output_dataset =np.log(Dis_transient_total) # Output dataset that will be used in training

#%%
# Preprocessing: 3. Fetch input data
# Preprocessing 3.1 Hysteretic Behaviour

# Hysteretic behaviors: HM1: 90, HM2: 27000, HM3: 27000
hysteresis = np.transpose(np.load('plot_hysteretic_data.npy'))

Hysteretic_ElemForc = hysteresis[0,1,:]  # Force
Hysteretic_NodeDisp = hysteresis[0,0,:]  # Displacement
Hysteretic_all=[]
for ii in range(54090): 
    tmp = np.transpose(np.asarray([Hysteretic_NodeDisp[:,ii], Hysteretic_ElemForc[:,ii]]))
    Hysteretic_all.append(tmp)
    
Hysteretic_all = np.asarray(Hysteretic_all)
Hysteretic_all = Hysteretic_all.reshape(54090,80,2)

Hysteretic_info_total=[]
for ii in range(NumberofGM):
    for jj in range(len(Hysteretic_all)):
        Hysteretic_info_total.append(Hysteretic_all[jj,:])

Hysteretic_info_total = np.asarray(Hysteretic_info_total)   
Input_hysteretic = Hysteretic_info_total.reshape(int(len(Hysteretic_all)*NumberofGM),80,2,1) # Input dataset that will be used in training (hysteretic)

del tmp, Hysteretic_ElemForc, Hysteretic_NodeDisp, ii, jj, hysteresis, Hysteretic_info_total

# Preprocessing 3.2 Ground motion information
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
for ii in range(len(RSN)):
    for jj in range(int(len(Input_hysteretic)/len(RSN))):
        Ground_info_total_b.append(Ground_info_total_b_temp[ii,:])
Ground_info_total_b = np.asarray(Ground_info_total_b)


Input_P  = np.log(Ground_info_total_b[:,7:10])  # Input ground motions, PGA, PGV, PGD
Input_MR = Ground_info_total_b[:,0:7]           # Input ground motions, M, R, Site
Input_Sa = np.log(Ground_info_total_b[:,10:])   # Input ground motions, Response spectrum

del Ground_info_total_b, Ground_info_total_b_temp, ii, Ground_info, Query_String, Magnitude, Site, temp_value, temp_RSN
###############################################################################
# End preprocess
###############################################################################
#%%
###############################################################################
# Construct DNN model
# Tensorflow and Keras are employed for this purpose
# Details of the DNN model is shown in 
###############################################################################

# Import Keras libraries
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

model.compile(loss='mean_squared_error', optimizer='Adam') # Adam optimizer is employed

###############################################################################
# End DNN model construction
###############################################################################
#%%
###############################################################################
# Training the DNN model
###############################################################################
# Data augmentation using the Keras libraries
# Height shift and vertical flip are employed
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(height_shift_range =0.2,
                                   vertical_flip=True,
                                   fill_mode='reflect')

# Usign the Data augmentation technique, hysteretic behaviors are varied
batches=0
Hys_batch = []
X1_P_batch = []
X1_MR_batch = []
X1_Sa_batch = []
y_batch = []
index = np.arange(0,len(Output_dataset))
for x_batch, index_batch in train_datagen.flow(Input_hysteretic, index, batch_size=3000):
    Hys_batch.append(x_batch)
    X1_P_batch.append(Input_P[index_batch,:])
    X1_MR_batch.append(Input_MR[index_batch,:])
    X1_Sa_batch.append(Input_Sa[index_batch,:])
    y_batch.append(Output_dataset[index_batch,])
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

# Train the DNN model
history_callback = model.fit([Hys_batch, X1_P_batch, X1_MR_batch, X1_Sa_batch], y_batch, batch_size = 512, epochs = Num_epochs)

# After training the DNN model, the results of the DNN model is saved as 'DNN_model_2019*.h5'    
# Displacement, velocity, and acceleration DNN models have been developed.