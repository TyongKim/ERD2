"""
This code provides how to use the DC rule.

Written by Taeyong Kim at Ajou University
taeyongkim@ajou.ac.kr
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
#%%
class DNNModel(nn.Module):
    def __init__(self):
        super(DNNModel, self).__init__()
        self.bn1 = nn.BatchNorm1d(14)
        self.fc1_1 = nn.Linear(14, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc1_2 = nn.Linear(16, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc1_3 = nn.Linear(16, 8)

        self.bn4 = nn.BatchNorm1d(220)
        self.fc2_1 = nn.Linear(220, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.fc2_2 = nn.Linear(32, 16)
        self.bn6 = nn.BatchNorm1d(16)
        self.fc2_3 = nn.Linear(16, 16)

        self.bn7 = nn.BatchNorm1d(24)
        self.fc3_1 = nn.Linear(24, 32)
        self.bn8 = nn.BatchNorm1d(32)
        self.fc3_2 = nn.Linear(32, 16)
        self.bn9 = nn.BatchNorm1d(16)
        self.fc3_3 = nn.Linear(16, 8)

        self.fc4_1 = nn.Linear(8, 4)
        self.fc4_2 = nn.Linear(8, 3)

    def forward(self, x1, x2):
        x1 = self.bn1(x1)
        x1 = F.relu(self.fc1_1(x1))
        x1 = self.bn2(x1)
        x1 = F.relu(self.fc1_2(x1))
        x1 = self.bn3(x1)
        x1 = F.relu(self.fc1_3(x1))

        x2 = self.bn4(x2)
        x2 = F.relu(self.fc2_1(x2))
        x2 = self.bn5(x2)
        x2 = F.relu(self.fc2_2(x2))
        x2 = self.bn6(x2)
        x2 = F.relu(self.fc2_3(x2))

        x = torch.cat((x1, x2), dim=1)
        x = self.bn7(x)
        x = F.relu(self.fc3_1(x))
        x = self.bn8(x)
        x = F.relu(self.fc3_2(x))
        x = self.bn9(x)
        x = F.relu(self.fc3_3(x))

        x4_1 = torch.sigmoid(self.fc4_1(x))
        x4_2 = torch.tanh(self.fc4_2(x))

        return torch.cat((x4_1, x4_2), dim=1)
    
def predict_SDOF(coeff, SDOF_response):
    
    c0 = coeff[:,0]*coeff[:,0]  # Longitudinal
    c1 = coeff[:,1]*coeff[:,1]  # Trnasverse 1st
    c2 = coeff[:,2]* coeff[:,2]+coeff[:,4]*coeff[:,4]
    c3 = coeff[:,3]* coeff[:,3]+coeff[:,5]* coeff[:,5]+coeff[:,6]* coeff[:,6]
    c4 = coeff[:,1]* coeff[:,4]
    c5 = coeff[:,1]* coeff[:,5]
    c6 = coeff[:,2]* coeff[:,6]+coeff[:,4]*coeff[:,5]
            
    Results_SDOF = np.sqrt(
                    c0*SDOF_response[:,1]**2+ # longitudinal
                    c1*SDOF_response[:,2]**2+
                    c2*SDOF_response[:,3]**2+
                    c3*SDOF_response[:,4]**2+
                    +2*c4*(SDOF_response[:,2]*SDOF_response[:,3])
                    +2*c5*(SDOF_response[:,2]*SDOF_response[:,4])
                    +2*c6*(SDOF_response[:,3]*SDOF_response[:,4])
                    )
    return Results_SDOF

def cal_coeff(coeff):
    
    c0 = coeff[:,0]*coeff[:,0]  # Longitudinal
    c1 = coeff[:,1]*coeff[:,1]  # Trnasverse 1st
    c2 = coeff[:,2]* coeff[:,2]+coeff[:,4]*coeff[:,4]
    c3 = coeff[:,3]* coeff[:,3]+coeff[:,5]* coeff[:,5]+coeff[:,6]* coeff[:,6]
    c4 = coeff[:,1]* coeff[:,4]
    c5 = coeff[:,1]* coeff[:,5]
    c6 = coeff[:,2]* coeff[:,6]+coeff[:,4]*coeff[:,5]
            

    return np.array([c0,c1,c2,c3,c4,c5,c6])


# Function to make predictions
def predict(model, input_struc1, input_GM):
    # Set the model to evaluation mode
    model.eval()
    
    # Disable gradient computation for inference
    with torch.no_grad():
        # Make predictions
        predictions = model(input_struc1, input_GM)
    
    return predictions

def func_cnt(arrs):
    zero_els = 0
    for arr in arrs:
        if arr == 0.0:
            zero_els = zero_els +1
    return zero_els
#%% Import data
# Load input and output data for DNN training  
DNN_input_GM_ver1 = np.load('./Results_DL/DL_info_GM.npy') 
DNN_input_STR = np.load('./Results_DL/DL_info_Displ_str.npy')
DNN_results_MDOF = np.load('./Results_DL/DL_info_MDOF_displ.npy')
DNN_results_MDOF = DNN_results_MDOF.reshape(len(DNN_results_MDOF),)
DNN_results_SDOF = np.load('./Results_DL/DL_info_SDOF_displ.npy')

DNN_analysis_results = np.c_[DNN_results_MDOF, DNN_results_SDOF[:, 0:4]]

# Prepare the data
input_struc1 = torch.tensor(DNN_input_STR, dtype=torch.float32)
input_GM = torch.tensor(DNN_input_GM_ver1, dtype=torch.float32)
mdof = torch.tensor(DNN_analysis_results, dtype=torch.float32)

# Create a custom dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_struc1, input_GM, mdof):
        self.input_struc1 = input_struc1
        self.input_GM = input_GM
        self.mdof = mdof
    
    def __len__(self):
        return len(self.input_struc1)
    
    def __getitem__(self, idx):
        return self.input_struc1[idx], self.input_GM[idx], self.mdof[idx]

# Instantiate the dataset and DataLoader
dataset = CustomDataset(input_struc1, input_GM, mdof)
dataloader2 = DataLoader(dataset, batch_size=16, shuffle=False)
#%% Load DNN model
model = DNNModel()
model.load_state_dict(torch.load('./DL_data/DNN_displ_wotime.pt'))
model.eval()

#%% Response spectrum method
# DC
Results_SDOF = [] # DNN-based prediction
for batch in dataloader2:
    input_struc1, input_GM, mdof = batch

    # Get predictions
    predictions = predict(model, input_struc1, input_GM)

    # Convert predictions to numpy array if needed
    predictions_numpy = predictions.numpy()
    abc1 = input_struc1.numpy()
    abc2 = input_GM.numpy()
    abc3 = mdof.numpy()
    abc4 = predictions_numpy

    tmp_Results_SDOF = predict_SDOF(predictions_numpy, mdof.numpy())
    Results_SDOF = np.r_[Results_SDOF,tmp_Results_SDOF]
    
# SRSS
Results_SRSS = []
for ii in range(len(DNN_results_SDOF)):
    tmp_srss = DNN_results_SDOF[ii,:]
    Results_SRSS.append(np.linalg.norm(tmp_srss))

Results_SRSS = np.asarray(Results_SRSS)    
    

# Errors
error_DNN_MSE_test =  np.mean((np.log(DNN_results_MDOF)
                               - np.log(Results_SDOF))**2)
error_SRSS_MSE_test =  np.mean((np.log(DNN_results_MDOF)
                                - np.log(Results_SRSS))**2)


error_DNN_MSE_mm =  np.mean((DNN_results_MDOF*1000 - Results_SDOF*1000)**2)
error_SRSS_MSE_mm = np.mean((DNN_results_MDOF*1000 - Results_SRSS*1000)**2)

error_DNN = np.mean(np.abs(DNN_results_MDOF - Results_SDOF)/DNN_results_MDOF)
error_SRSS = np.mean(np.abs(DNN_results_MDOF - Results_SRSS)/DNN_results_MDOF) 

print('Training_dataset error: MSE')
print(error_DNN_MSE_test)
print(error_SRSS_MSE_test)

print('Training_dataset error: MSE; mm units')
print(error_DNN_MSE_mm)
print(error_SRSS_MSE_mm)

print('Training_dataset error: RE')
print(error_DNN)
print(error_SRSS)

print('Training dataset error: RE max')
print(np.max(np.abs(DNN_results_MDOF - Results_SDOF)/DNN_results_MDOF))     
print(np.max(np.abs(DNN_results_MDOF - Results_SRSS)/DNN_results_MDOF)) 
