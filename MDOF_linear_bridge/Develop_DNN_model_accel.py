"""
The script shows how to train the DNN model for the DC rule.

Written by Taeyong Kim at Ajou University
taeyongkim@ajou.ac.kr
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
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
        self.fc4_3 = nn.Linear(8, 1)

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
        x4_3 = torch.sigmoid(self.fc4_3(x))

        return torch.cat((x4_1, x4_2, x4_3), dim=1)


# Custom loss function
def custom_loss(y_pred, inp_1):
    MDOF = inp_1[:, 0]

    c0 = y_pred[:, 0] ** 2
    c1 = y_pred[:, 1] ** 2
    c2 = y_pred[:, 2] ** 2 + y_pred[:, 4] ** 2
    c3 = y_pred[:, 3] ** 2 + y_pred[:, 5] ** 2 + y_pred[:, 6] ** 2
    c4 = y_pred[:, 1] * y_pred[:, 4]
    c5 = y_pred[:, 1] * y_pred[:, 5]
    c6 = y_pred[:, 2] * y_pred[:, 6] + y_pred[:, 4] * y_pred[:, 5]
    c7 = y_pred[:, 7] ** 2

    SDOF = torch.sqrt(
        (inp_1[:, 1] ** 2) * c0 +
        (inp_1[:, 2] ** 2) * c1 +
        (inp_1[:, 3] ** 2) * c2 +
        (inp_1[:, 4] ** 2) * c3 +
        2 * c4 * inp_1[:, 2] * inp_1[:, 3] +
        2 * c5 * inp_1[:, 2] * inp_1[:, 4] +
        2 * c6 * inp_1[:, 3] * inp_1[:, 4] +
        (inp_1[:, 5] ** 2) * c7 
    )

    MDOF = torch.log(MDOF)
    SDOF = torch.log(SDOF)

    loss = torch.mean((MDOF - SDOF) ** 2)
    return loss
    
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


#%% Import data
# Load input and output data for DNN training  
DNN_input_GM_ver1 = np.load('./Results_DL/DL_info_GM.npy') 
DNN_input_STR = np.load('./Results_DL/DL_info_Displ_str.npy')
DNN_results_MDOF = np.load('./Results_DL/DL_info_MDOF_accel.npy')
DNN_results_MDOF = DNN_results_MDOF.reshape(len(DNN_results_MDOF),)
DNN_results_SDOF = np.load('./Results_DL/DL_info_SDOF_accel.npy')

DNN_input_accel_GM = np.load('./Results_DL/DL_info_Accel_GM.npy')
DNN_input_accel_STR = np.load('./Results_DL/DL_info_Accel_str.npy')

DNN_input_accel_EQ = []
for ii in range(len(DNN_input_accel_GM)):
    tmp = DNN_input_accel_STR[ii,0]*DNN_input_accel_GM[ii]*9.81
    DNN_input_accel_EQ.append(tmp)

DNN_input_accel_EQ = np.asarray(DNN_input_accel_EQ)
DNN_analysis_results = np.c_[DNN_results_MDOF, DNN_results_SDOF[:, 0:4], DNN_input_accel_EQ]
DNN_results_MDOF = DNN_results_MDOF.reshape(len(DNN_results_MDOF),)

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
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Instantiate the model
model = DNNModel()
# Training the model with L2 regularization
def train_model(model, dataloader, num_epochs, learning_rate=0.001, l2_lambda=0.0):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    
    for epoch in range(num_epochs):
        model.train()
        
        epoch_loss = 0
        for batch in dataloader:
            input_struc1, input_GM, mdof = batch
            
            for repeate in range(5): 
                # Forward pass
                outputs = model(input_struc1, input_GM)
                loss = custom_loss(outputs, mdof)
    
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}')

# Train the model
train_model(model, dataloader, num_epochs=100, learning_rate=0.001, l2_lambda=0.0005)

