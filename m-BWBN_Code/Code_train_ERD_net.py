"""
This script is to develop a DNN model to predict structural responses under an
earthquake. In order to perform this code, the ERD-base is required. 
The database can be either developed by users or downloaded at http://ERD2.snu.ac.kr

Moreover, although this script is written to develop a DNN model for the peak 
transient displacement of structural systems, other structural responses
such as acceleration and velocity are also employed as an output of a DNN 
model.

The deep learning model is developed based on the TensorFlow and Keras. Please
download such libraries.

The code is developed by Taeyong Kim from the Ajou University
taeyongkim@ajou.ac.kr

March 20, 2023
"""

# Import libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import sqlite3

# version check
print(tf.__version__)
print(keras.__version__)
#%% DNN model architecture
def create_model():
    input_CNN = keras.Input(shape=(241, 2, 1))
    conv1_1 = keras.layers.Conv2D(filters=32, kernel_size=[2, 2], padding='SAME')(input_CNN)
    conv1_2 = keras.layers.Conv2D(filters=32, kernel_size=[4, 2], padding='SAME')(input_CNN)
    conv1_3 = keras.layers.Conv2D(filters=32, kernel_size=[8, 2], padding='SAME')(input_CNN)
    conv1_4 = keras.layers.Conv2D(filters=32, kernel_size=[16, 2], padding='SAME')(input_CNN)

    conv1_5 = keras.layers.concatenate([conv1_1, conv1_2, conv1_3, conv1_4],axis=3)
    conv1_5 = tf.nn.tanh(conv1_5)

    conv2_1 = keras.layers.Conv2D(filters=32, kernel_size=[2, 2], padding='SAME')(conv1_5)
    conv2_2 = keras.layers.Conv2D(filters=32, kernel_size=[4, 2], padding='SAME')(conv1_5)
    conv2_3 = keras.layers.Conv2D(filters=32, kernel_size=[8, 2], padding='SAME')(conv1_5)
    conv2_4 = keras.layers.Conv2D(filters=32, kernel_size=[16, 2], padding='SAME')(conv1_5)

    conv2_5 = keras.layers.concatenate([conv2_1, conv2_2, conv2_3, conv2_4],axis=3)
    conv2_5 = keras.layers.Activation(activation='relu')(conv2_5)
    conv2_5 = keras.layers.BatchNormalization()(conv2_5)

    
    conv3_1_1 = keras.layers.Conv2D(filters=16, kernel_size=[2, 2], padding='SAME')(conv2_5)
    conv3_1_1 = keras.layers.MaxPool2D(pool_size=(2, 1), padding='valid')(conv3_1_1)
    conv3_1_1 = keras.layers.BatchNormalization()(conv3_1_1)
    conv3_1_2 = keras.layers.Conv2D(filters=16, kernel_size=[2, 2], padding='SAME')(conv3_1_1)
    conv3_1_2 = keras.layers.MaxPool2D(pool_size=(2, 1), padding='valid')(conv3_1_2)   
    conv3_1_2 = keras.layers.BatchNormalization()(conv3_1_2)    
    conv3_1_3 = keras.layers.Conv2D(filters=32, kernel_size=[2, 2], padding='SAME')(conv3_1_2)
    conv3_1_3 = keras.layers.MaxPool2D(pool_size=(2, 1), padding='valid')(conv3_1_3)    
    
    conv3_2_1 = keras.layers.Conv2D(filters=16, kernel_size=[4, 2], padding='SAME')(conv2_5)
    conv3_2_1 = keras.layers.MaxPool2D(pool_size=(2, 1), padding='valid')(conv3_2_1)
    conv3_2_1 = keras.layers.BatchNormalization()(conv3_2_1)
    conv3_2_2 = keras.layers.Conv2D(filters=16, kernel_size=[4, 2], padding='SAME')(conv3_2_1)
    conv3_2_2 = keras.layers.MaxPool2D(pool_size=(2, 1), padding='valid')(conv3_2_2)   
    conv3_2_2 = keras.layers.BatchNormalization()(conv3_2_2)    
    conv3_2_3 = keras.layers.Conv2D(filters=32, kernel_size=[4, 2], padding='SAME')(conv3_2_2)
    conv3_2_3 = keras.layers.MaxPool2D(pool_size=(2, 1), padding='valid')(conv3_2_3)      
    
    conv3_3_1 = keras.layers.Conv2D(filters=16, kernel_size=[8, 2], padding='SAME')(conv2_5)
    conv3_3_1 = keras.layers.MaxPool2D(pool_size=(2, 1), padding='valid')(conv3_3_1)
    conv3_3_1 = keras.layers.BatchNormalization()(conv3_3_1)
    conv3_3_2 = keras.layers.Conv2D(filters=16, kernel_size=[8, 2], padding='SAME')(conv3_3_1)
    conv3_3_2 = keras.layers.MaxPool2D(pool_size=(2, 1), padding='valid')(conv3_3_2)   
    conv3_3_2 = keras.layers.BatchNormalization()(conv3_3_2)    
    conv3_3_3 = keras.layers.Conv2D(filters=32, kernel_size=[8, 2], padding='SAME')(conv3_3_2)
    conv3_3_3 = keras.layers.MaxPool2D(pool_size=(2, 1), padding='valid')(conv3_3_3)      
    
    conv3_4_1 = keras.layers.Conv2D(filters=16, kernel_size=[16, 2], padding='SAME')(conv2_5)
    conv3_4_1 = keras.layers.MaxPool2D(pool_size=(2, 1), padding='valid')(conv3_4_1)
    conv3_4_1 = keras.layers.BatchNormalization()(conv3_4_1)
    conv3_4_2 = keras.layers.Conv2D(filters=16, kernel_size=[16, 2], padding='SAME')(conv3_4_1)
    conv3_4_2 = keras.layers.MaxPool2D(pool_size=(2, 1), padding='valid')(conv3_4_2)   
    conv3_4_2 = keras.layers.BatchNormalization()(conv3_4_2)    
    conv3_4_3 = keras.layers.Conv2D(filters=32, kernel_size=[16, 2], padding='SAME')(conv3_4_2)
    conv3_4_3 = keras.layers.MaxPool2D(pool_size=(2, 1), padding='valid')(conv3_4_3)      
    
    conv4_1_1 = keras.layers.Flatten()(conv3_1_3)
    conv4_2_1 = keras.layers.Flatten()(conv3_2_3)
    conv4_3_1 = keras.layers.Flatten()(conv3_3_3)
    conv4_4_1 = keras.layers.Flatten()(conv3_4_3)

    conv4_1_2 = keras.layers.BatchNormalization()(conv4_1_1)        
    conv4_1_2 = keras.layers.Dense(units=128, activation='relu')(conv4_1_2)
    conv4_1_3 = keras.layers.BatchNormalization()(conv4_1_2)        
    conv4_1_3 = keras.layers.Dense(units=32, activation='relu')(conv4_1_3)

    conv4_2_2 = keras.layers.BatchNormalization()(conv4_2_1)        
    conv4_2_2 = keras.layers.Dense(units=128, activation='relu')(conv4_2_2)
    conv4_2_3 = keras.layers.BatchNormalization()(conv4_2_2)        
    conv4_2_3 = keras.layers.Dense(units=32, activation='relu')(conv4_2_3)

    conv4_3_2 = keras.layers.BatchNormalization()(conv4_3_1)        
    conv4_3_2 = keras.layers.Dense(units=128, activation='relu')(conv4_3_2)
    conv4_3_3 = keras.layers.BatchNormalization()(conv4_3_2)        
    conv4_3_3 = keras.layers.Dense(units=32, activation='relu')(conv4_3_3)        

    conv4_4_2 = keras.layers.BatchNormalization()(conv4_4_1)        
    conv4_4_2 = keras.layers.Dense(units=128, activation='relu')(conv4_4_2)
    conv4_4_3 = keras.layers.BatchNormalization()(conv4_4_2)        
    conv4_4_3 = keras.layers.Dense(units=32, activation='relu')(conv4_4_3)  

    conv5_1 = keras.layers.concatenate([conv4_1_3, conv4_2_3, conv4_3_3, conv4_4_3])
    conv5_1 = keras.layers.BatchNormalization()(conv5_1)        
    conv5_1 = keras.layers.Dense(units=64, activation='relu')(conv5_1)
    conv5_1 = keras.layers.BatchNormalization()(conv5_1)        
    conv5_1 = keras.layers.Dense(units=128, activation='relu')(conv5_1)


    # Earthquake information - MR    
    input_EQ_MR = keras.Input(shape=(7,))
    MR_layer1_1 = keras.layers.BatchNormalization()(input_EQ_MR)
    MR_layer1_1 = keras.layers.Dense(units=16)(MR_layer1_1)
    MR_layer1_1 = keras.layers.Activation(activation='relu')(MR_layer1_1)
    MR_layer1_2 = keras.layers.BatchNormalization()(MR_layer1_1)
    MR_layer1_2 = keras.layers.Dense(units=8)(MR_layer1_2)
    MR_layer1_2 = keras.layers.Activation(activation='relu')(MR_layer1_2)    
    MR_layer1_3 = keras.layers.BatchNormalization()(MR_layer1_2)
    MR_layer1_3 = keras.layers.Dense(units=8)(MR_layer1_3)
    MR_layer1_3 = keras.layers.Activation(activation='relu')(MR_layer1_3)  
    
    MR_layer3_1 = keras.layers.concatenate([conv5_1, MR_layer1_3])
    MR_layer3_1 = keras.layers.BatchNormalization()(MR_layer3_1)
    MR_layer3_1 = keras.layers.Dense(units=32, kernel_regularizer=keras.regularizers.l2(0.001))(MR_layer3_1)
    MR_layer3_1 = keras.layers.Activation(activation='relu')(MR_layer3_1)
    MR_layer3_2 = keras.layers.BatchNormalization()(MR_layer3_1)
    MR_layer3_2 = keras.layers.Dense(units=16, kernel_regularizer=keras.regularizers.l2(0.001))(MR_layer3_2)
    MR_layer3_2 = keras.layers.Activation(activation='relu')(MR_layer3_2)
    MR_layer3_3 = keras.layers.BatchNormalization()(MR_layer3_2)
    MR_layer3_3 = keras.layers.Dense(units=16, kernel_regularizer=keras.regularizers.l2(0.001))(MR_layer3_3)
    MR_layer3_3 = keras.layers.Activation(activation='relu')(MR_layer3_3)   

    # Earthquake information - Sa
    input_EQ_Sa = keras.Input(shape=(110,))
    Sa_layer1_1 = keras.layers.BatchNormalization()(input_EQ_Sa)
    Sa_layer1_1 = keras.layers.Dense(units=64)(Sa_layer1_1)
    Sa_layer1_1 = keras.layers.Activation(activation='relu')(Sa_layer1_1)
    Sa_layer1_2 = keras.layers.BatchNormalization()(Sa_layer1_1)
    Sa_layer1_2 = keras.layers.Dense(units=32)(Sa_layer1_2)
    Sa_layer1_2 = keras.layers.Activation(activation='relu')(Sa_layer1_2)    
    Sa_layer1_3 = keras.layers.BatchNormalization()(Sa_layer1_2)
    Sa_layer1_3 = keras.layers.Dense(units=32)(Sa_layer1_3)
    Sa_layer1_3 = keras.layers.Activation(activation='relu')(Sa_layer1_3)  

    Sa_layer3_1 = keras.layers.concatenate([conv5_1, Sa_layer1_3])
    Sa_layer3_1 = keras.layers.BatchNormalization()(Sa_layer3_1)
    Sa_layer3_1 = keras.layers.Dense(units=64, kernel_regularizer=keras.regularizers.l2(0.001))(Sa_layer3_1)
    Sa_layer3_1 = keras.layers.Activation(activation='relu')(Sa_layer3_1)
    Sa_layer3_2 = keras.layers.BatchNormalization()(Sa_layer3_1)
    Sa_layer3_2 = keras.layers.Dense(units=32, kernel_regularizer=keras.regularizers.l2(0.001))(Sa_layer3_2)
    Sa_layer3_2 = keras.layers.Activation(activation='relu')(Sa_layer3_2)  
    Sa_layer3_3 = keras.layers.BatchNormalization()(Sa_layer3_2)
    Sa_layer3_3 = keras.layers.Dense(units=32, kernel_regularizer=keras.regularizers.l2(0.001))(Sa_layer3_3)
    Sa_layer3_3 = keras.layers.Activation(activation='relu')(Sa_layer3_3)  

    # Earthquake information - Pg
    input_EQ_PG = keras.Input(shape=(6,))
    PG_layer1_1 = keras.layers.BatchNormalization()(input_EQ_PG)
    PG_layer1_1 = keras.layers.Dense(units=16)(PG_layer1_1)
    PG_layer1_1 = keras.layers.Activation(activation='relu')(PG_layer1_1)
    PG_layer1_2 = keras.layers.BatchNormalization()(PG_layer1_1)
    PG_layer1_2 = keras.layers.Dense(units=8)(PG_layer1_2)
    PG_layer1_2 = keras.layers.Activation(activation='relu')(PG_layer1_2)    
    PG_layer1_3 = keras.layers.BatchNormalization()(PG_layer1_2)
    PG_layer1_3 = keras.layers.Dense(units=8)(PG_layer1_3)
    PG_layer1_3 = keras.layers.Activation(activation='relu')(PG_layer1_3)    

    PG_layer3_1 = keras.layers.concatenate([conv5_1, PG_layer1_3])
    PG_layer3_1 = keras.layers.BatchNormalization()(PG_layer3_1)
    PG_layer3_1 = keras.layers.Dense(units=32, kernel_regularizer=keras.regularizers.l2(0.001))(PG_layer3_1)
    PG_layer3_1 = keras.layers.Activation(activation='relu')(PG_layer3_1)
    PG_layer3_2 = keras.layers.BatchNormalization()(PG_layer3_1)
    PG_layer3_2 = keras.layers.Dense(units=16, kernel_regularizer=keras.regularizers.l2(0.001))(PG_layer3_2)
    PG_layer3_2 = keras.layers.Activation(activation='relu')(PG_layer3_2)   
    PG_layer3_3 = keras.layers.BatchNormalization()(PG_layer3_2)
    PG_layer3_3 = keras.layers.Dense(units=16, kernel_regularizer=keras.regularizers.l2(0.001))(PG_layer3_3)
    PG_layer3_3 = keras.layers.Activation(activation='relu')(PG_layer3_3)       

    FC1 = keras.layers.concatenate([conv5_1, MR_layer3_3, Sa_layer3_3, PG_layer3_3])
    FC1 = keras.layers.BatchNormalization()(FC1)
    FC1 = keras.layers.Dense(units=1024, kernel_regularizer=keras.regularizers.l2(0.0005))(FC1)
    FC1 = keras.layers.Activation(activation='relu')(FC1)

    FC2 = keras.layers.BatchNormalization()(FC1)
    FC2 = keras.layers.Dense(units=512, kernel_regularizer=keras.regularizers.l2(0.0005))(FC2)
    FC2 = keras.layers.Activation(activation='relu')(FC2)

    FC3 = keras.layers.BatchNormalization()(FC2)
    FC3 = keras.layers.Dense(units=256, kernel_regularizer=keras.regularizers.l2(0.0005))(FC3)
    FC3 = keras.layers.Activation(activation='relu')(FC3)

    FC4 = keras.layers.BatchNormalization()(FC3)
    FC4 = keras.layers.Dense(units=32, kernel_regularizer=keras.regularizers.l2(0.0005))(FC4)
    FC4 = keras.layers.Activation(activation='relu')(FC4)

    FC5 = keras.layers.BatchNormalization()(FC4)
    FC5 = keras.layers.Dense(units=1)(FC5)
    FC5 = keras.layers.Activation(activation='linear')(FC5)

    return keras.Model(inputs=[input_CNN, input_EQ_MR, input_EQ_Sa, input_EQ_PG], outputs=FC5)


model = create_model()
model.summary()
model.compile(
    optimizer=tf.optimizers.Adam(),
    loss='mean_squared_error')

#%% Import dataset
# Record sequence number used in the ERD-base
conn2 = sqlite3.connect("SDDB_mBWBN_ver1.0.db")
db2 = conn2.cursor() 

# Find number of GM
Query_String = "select RSN "\
               "FROM Ground_motion "\
               "    order by RSN "\

db2.execute(Query_String) 
input_db  = db2.fetchall()
RSN = np.asarray(input_db)
RSN = RSN.reshape(len(RSN),)
 
conn2.close()
# Select test set and training set (Using 80% for training: total 1199 among 1499)
import random
SEED = 117

random.seed(SEED)
random.shuffle(RSN)

# Discretize train and test dataset
RSN_training = RSN[0:np.int32(len(RSN)*0.8)] # total of 1,199 datasets
RSN_test = RSN[np.int32(len(RSN)*0.8):,] 
del RSN    


# Import function to fetch the dataset
from import_dataset import import_dataset

Num_train_GM=2; # Number of ground motions used in training

# Parameters for training
training_epoch = 5
batch_size_custom = 512

# Datatset training
[x_data2_1, x_data2_2, x_data2_3, x_data1, 
 y_data] = import_dataset(RSN_training, Num_train_GM)

#%% Train the DNN model
history_callback = model.fit([x_data1, x_data2_1, x_data2_2, x_data2_3], 
                             y_data, 
                             batch_size = batch_size_custom, 
                             epochs = training_epoch, shuffle=True)

model.save('ERD_net_displ.h5') # Save the trained model
