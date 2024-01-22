###############################################################################
# This code is developed by Prof. Taeyong Kim at Ajou University              #
# taeyongkim@ajou.ac.kr                                                       #
# Jan 20, 2024                                                                #
###############################################################################

# Import libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
# version check
print(tf.__version__)
#%% Investigate results
# Custom loss function
def custom_Loss_with_input(inp_1):
    def loss(y_true, y_pred):

        MDOF = inp_1[:,0]
        
        c1 = tf.math.multiply(y_pred[:,0], y_pred[:,0])
        c2 = tf.math.multiply(y_pred[:,1], y_pred[:,1])+tf.math.multiply(y_pred[:,3], y_pred[:,3])
        c3 = tf.math.multiply(y_pred[:,2], y_pred[:,2])+tf.math.multiply(y_pred[:,4], y_pred[:,4])+tf.math.multiply(y_pred[:,5], y_pred[:,5])
        c4 = tf.math.multiply(y_pred[:,0], y_pred[:,3])
        c5 = tf.math.multiply(y_pred[:,0], y_pred[:,4])
        c6 = tf.math.multiply(y_pred[:,1], y_pred[:,5])+tf.math.multiply(y_pred[:,3], y_pred[:,4])
        
        SDOF = tf.math.sqrt(
                tf.math.multiply( tf.math.square(inp_1[:,1]), c1)+
                tf.math.multiply( tf.math.square(inp_1[:,2]), c2)+
                tf.math.multiply( tf.math.square(inp_1[:,3]), c3)
                +2* tf.math.multiply(tf.math.multiply(c4,inp_1[:,1]),inp_1[:,2])
                +2* tf.math.multiply(tf.math.multiply(c5,inp_1[:,1]),inp_1[:,3])
                +2* tf.math.multiply(tf.math.multiply(c6,inp_1[:,2]),inp_1[:,3])

                )  
        
        MDOF = tf.math.log(MDOF)
        SDOF = tf.math.log(SDOF)

        result = tf.math.reduce_mean(tf.square(MDOF-SDOF))
        return result
    return loss
        
def predict_SDOF(coeff, SDOF_response):
    
    c1 = coeff[:,0]*coeff[:,0]
    c2 = coeff[:,1]* coeff[:,1]+coeff[:,3]*coeff[:,3]
    c3 = coeff[:,2]* coeff[:,2]+coeff[:,4]* coeff[:,4]+coeff[:,5]* coeff[:,5]
    c4 = coeff[:,0]* coeff[:,3]
    c5 = coeff[:,0]* coeff[:,4]
    c6 = coeff[:,1]* coeff[:,5]+coeff[:,3]*coeff[:,4]
        
    
    
    Results_SDOF = np.sqrt(
                    c1*SDOF_response[:,1]**2+
                    c2*SDOF_response[:,2]**2+
                    c3*SDOF_response[:,3]**2+
                    +2*c4*(SDOF_response[:,1]*SDOF_response[:,2])
                    +2*c5*(SDOF_response[:,1]*SDOF_response[:,3])
                    +2*c6*(SDOF_response[:,2]*SDOF_response[:,3])
                    )

    return Results_SDOF

#%% DNN model architecture
def define_model():

    Input_struc1 = keras.Input(shape=(11,))
    
    model1_2 = keras.layers.BatchNormalization()(Input_struc1)    
    model1_2 = keras.layers.Dense(units=32)(model1_2)
    model1_2 = keras.layers.Activation(activation='relu')(model1_2)
    model1_2 = keras.layers.BatchNormalization()(model1_2)    
    model1_2 = keras.layers.Dense(units=16)(model1_2)
    model1_2 = keras.layers.Activation(activation='relu')(model1_2)
    model1_2 = keras.layers.BatchNormalization()(model1_2)    
    model1_2 = keras.layers.Dense(units=8)(model1_2)
    model1_2 = keras.layers.Activation(activation='relu')(model1_2)

    
    Input_GM = keras.Input(shape=(123,))
    model2 = keras.layers.BatchNormalization()(Input_GM)    
    model2 = keras.layers.Dense(units=16, kernel_regularizer=keras.regularizers.l2(0.002))(model2)
    model2 = keras.layers.Activation(activation='relu')(model2)
    model2 = keras.layers.BatchNormalization()(model2)    
    model2 = keras.layers.Dense(units=16, kernel_regularizer=keras.regularizers.l2(0.002))(model2)
    model2 = keras.layers.Activation(activation='relu')(model2)
    model2 = keras.layers.BatchNormalization()(model2)        
    model2 = keras.layers.Dense(units=8, kernel_regularizer=keras.regularizers.l2(0.002))(model2)
    model2 = keras.layers.Activation(activation='relu')(model2)
    
    model3 = keras.layers.concatenate([model1_2, model2], axis = 1)    
    
    model3 = keras.layers.Flatten()(model3)
    model3 = keras.layers.BatchNormalization()(model3)        
    model3 = keras.layers.Dense(units=32, kernel_regularizer=keras.regularizers.l2(0.002))(model3)
    model3 = keras.layers.Activation(activation='relu')(model3) 
    model3 = keras.layers.BatchNormalization()(model3)            
    model3 = keras.layers.Dense(units=16, kernel_regularizer=keras.regularizers.l2(0.002))(model3)
    model3 = keras.layers.Activation(activation='relu')(model3)  
    model3 = keras.layers.BatchNormalization()(model3)
    model3 = keras.layers.Dense(units=8, kernel_regularizer=keras.regularizers.l2(0.002))(model3)
    model3 = keras.layers.Activation(activation='relu')(model3)  
    model3 = keras.layers.BatchNormalization()(model3)   

    model3_1 = keras.layers.Dense(units=3)(model3)
    model3_1 = keras.layers.Activation(activation='sigmoid')(model3_1)      
    model3_2 = keras.layers.Dense(units=3)(model3)
    model3_2 = keras.layers.Activation(activation='tanh')(model3_2)    

    model3 = keras.layers.concatenate([model3_1, model3_2], axis = 1)    
    
    return keras.Model(inputs=[Input_struc1, Input_GM], outputs=model3) 

model = define_model()
model.summary()

#%% Train the DNN model
batch_size = 1000
epochs_num = 2

# Load dataset
DNN_input1_GM_train = np.load('./DNN_results/DNN_input1_GM_train.npy') # already log
DNN_input1_str_train = np.load('./DNN_results/DNN_input2_str_train.npy') # already log
DNN_input1_response_train = np.load('./DNN_results/DNN_input3_response_train.npy', 
                                    allow_pickle=True) # meter

tmp_DNN_GM = []
tmp_DNN_str = []
tmp_DNN_response = []
# Becasue each response has multiple IDR and one drift ratio
for ii in range(len(DNN_input1_response_train)):
    tmp_response=DNN_input1_response_train[ii]
    
    for jj in range(len(tmp_response)):
        
        if jj == int(len(tmp_response)-1):
            tmp_DNN_str.append(np.r_[DNN_input1_str_train[ii,:],
                                 1,0] )                
        else:
            tmp_DNN_str.append(np.r_[DNN_input1_str_train[ii,:],
                                 0,np.log((jj+1)/(len(tmp_response)-1))] )

        tmp_DNN_GM.append(DNN_input1_GM_train[ii,:])
        tmp_DNN_response.append(tmp_response[jj,:])
        
tmp_DNN_GM = np.asarray(tmp_DNN_GM)        
tmp_DNN_str = np.asarray(tmp_DNN_str)        
tmp_DNN_response = np.asarray(tmp_DNN_response)        

del DNN_input1_GM_train, DNN_input1_str_train, DNN_input1_response_train

# Train DNN model
for ii in range(epochs_num): # Number of epochs
    
    # Chop dataset based on the batch size
    tmp3_DNN_fake_output = np.zeros([batch_size, 6])
    for jj in range(int(len(tmp_DNN_response)/batch_size)):
        tmp3_DNN_GM = tmp_DNN_GM[batch_size*jj:batch_size*(jj+1),:]
        tmp3_DNN_str = tmp_DNN_str[batch_size*jj:batch_size*(jj+1),:]
        tmp3_DNN_response = np.abs(tmp_DNN_response[batch_size*jj:batch_size*(jj+1),:])
        tmp3_DNN_response2 = tf.convert_to_tensor(tmp3_DNN_response, dtype=tf.float32) 
        
        model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(),
        loss=custom_Loss_with_input(tmp3_DNN_response2))    

        history_callback = model.fit([tmp3_DNN_str, tmp3_DNN_GM], tmp3_DNN_fake_output,
                             batch_size = batch_size,
                             epochs = 10, 
                             shuffle=False)

# Save the model
tmp_filename = './DNN_results/tmp_DC_Model.h5'
model.save(tmp_filename)
