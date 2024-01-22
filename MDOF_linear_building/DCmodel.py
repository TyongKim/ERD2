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

#%% Train the DNN model
batch_size = 1000
epochs_num = 100

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

# Load model
model = keras.models.load_model("./DNN_results/DC_Model.h5", 
                                custom_objects={"loss": custom_Loss_with_input(1)})
# Predict using the DC model
predict_coeff = model.predict([tmp_DNN_str, tmp_DNN_GM])

# Predict the value
Response_MDOF = tmp_DNN_response[:,0]      
Results_SDOF_DC = predict_SDOF(predict_coeff, np.abs(tmp_DNN_response))
Results_SDOF_SRSS = np.sqrt(tmp_DNN_response[:,1]**2+tmp_DNN_response[:,2]**2+tmp_DNN_response[:,3]**2)

# Error
Error_DC = np.abs(Response_MDOF-Results_SDOF_DC)/Response_MDOF*100 # Relative error %
Error_SRSS = np.abs(Response_MDOF-Results_SDOF_SRSS)/Response_MDOF*100 # Relative error %

print(np.mean(Error_DC))
print(np.mean(Error_SRSS))
