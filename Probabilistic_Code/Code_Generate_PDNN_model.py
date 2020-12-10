"""

This script is to develop a P-DNN model to probabilistically predict structural 
responses under an earthquake. In order to perform this code, the "Seismic 
Demand Database (SDDB)" is required. The database can be downloaded 
at http://ERD2.snu.ac.kr.

Moreover, although this script is written to develop a P-DNN model for the 
peak transient displacement of a structural system, other structural responses
such as acceleration and velocity are also employed as an output of a DNN 
model.

The deep learning model is developed based on the TensorFlow and Keras. Please
download the corresponding libraries.

The code is developed by Taeyong Kim from the Seoul National University
chs5566@snu.ac.kr

"""

# Basic libraries
import numpy as np
import sqlite3
import random
import tensorflow as tf
import keras

# In a local computer: set very small number, and in super computer 
# use the entire training ground motions
NumberofGM = 2 # In super computer, 1199
Num_epochs = 10 # Number of epochs for training

# version check
print(tf.__version__)
print(keras.__version__)

# If using MAC, this might be helpful
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
# Construct P-DNN model
# The model is an extension of the DNN model. Thus, model parameters and 
# architecture of the DNN model is impoted to construct the P-DNN model.
# Tensorflow and Keras are employed for this purpose
# Details of the DNN model is shown by the command "model.summary()"
###############################################################################

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

# Load DNN model
model = keras.models.load_model('DNN_model_2019_displ.h5') # The trained DNN model

# Discard the original merged layers
for ii in range(16):
    model.layers.pop()

# Import Keras libraries
from keras.layers import Dense
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate

# The final merged layers for the P-DNN model
def fianl_ann(merged_conv1, name):

    # Make convolute model 
    merged_conv1 = BatchNormalization(axis=-1, name=name+"batch1")(merged_conv1)
    
    merged_conv1 = Dense(units = 512, name=name+"new1", activation='relu')(merged_conv1)
    
    merged_conv1 = BatchNormalization(axis=-1, name=name+"batch2")(merged_conv1)
    merged_conv1 = Dense(units = 512, name=name+"new2", activation='relu')(merged_conv1)
    
    merged_conv1 = BatchNormalization(axis=-1, name=name+"batch3")(merged_conv1)
    merged_conv1 = Dense(units = 256, name=name+"new3", activation='relu')(merged_conv1)
    
    merged_conv1 = BatchNormalization(axis=-1, name=name+"batch4")(merged_conv1)
    merged_conv1_result = Dense(units = 256, name=name+"new4_1", activation='relu')(merged_conv1)
    merged_conv1_aleato = Dense(units = 256, name=name+"new4_2", activation='relu')(merged_conv1)
    
    merged_conv2_result = Dense(units = 128, name=name+"new5_1", activation='relu')(merged_conv1_result)
    merged_conv2_aleato = Dense(units = 128, name=name+"new5_2", activation='relu')(merged_conv1_aleato)

    merged_conv3_result = Dense(units = 64, name=name+"new6_1", activation='relu')(merged_conv2_result)
    merged_conv3_aleato = Dense(units = 64, name=name+"new6_2", activation='relu')(merged_conv2_aleato)    

    merged_conv4_result = Dense(units = 32, name=name+"new7_1", activation='relu')(merged_conv3_result)
    merged_conv4_aleato = Dense(units = 32, name=name+"new7_2", activation='relu')(merged_conv3_aleato) 

    merged_conv5_result = Dense(units = 1, name=name+"new8_1", activation='linear')(merged_conv4_result)
    merged_conv5_aleato = Dense(units = 1, name=name+"new8_2", activation='linear')(merged_conv4_aleato)
    
    return merged_conv5_result, merged_conv5_aleato

# To connect the DNN model and the newly defined merged layers in the 'final_ann'
hysteretic = model.layers[119].output
gm_p_p = model.layers[150].output
gm_sa_sa = model.layers[151].output
gm_mr_mr = model.layers[152].output

merged_conv7 = concatenate([hysteretic, gm_p_p, gm_sa_sa, gm_mr_mr], name = 'convolute7')
merged_conv7_result, merged_conv7_aleato = fianl_ann(merged_conv7, "PGA_Sa_MR")

# Compiling the CNN
model2 = Model(inputs = model.input, outputs = [merged_conv7_aleato, merged_conv7_result])
del model # delete DNN model in order to confused with teh P-DNN model

# Assign the loss function to the model
# Mean and variance wich are predicted from the P-DNN model
linear_loss = mse_var_wrapper(merged_conv7_result)
aleato_loss = mse_lin_wrapper(merged_conv7_aleato)

# If you want to transfer learning the following code might be helpful,
# if not, ignore the codes.
for layer in model2.layers[:153]:
    layer.trainable = False

# Compile the model    
model2.compile(loss={'PGA_Sa_MRnew8_1':aleato_loss , 'PGA_Sa_MRnew8_2': linear_loss }, 
              loss_weights={'PGA_Sa_MRnew8_1': .5, 'PGA_Sa_MRnew8_2': .5}, optimizer='Adam')

###############################################################################
# P-DNN model is constructed whose contents are included in 'model2' variable
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
history_callback = model2.fit([Hys_batch, X1_P_batch, X1_MR_batch, X1_Sa_batch], 
                              [y_batch,y_batch], batch_size = 512, epochs = Num_epochs)

# After training the DNN model, the results of the DNN model is saved as 
# 'P_DNN_model_2019*.h5'    
# Displacement, velocity, and acceleration P-DNN models have been developed.