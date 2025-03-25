# Software introduction
This software is designed to predict the responses (displacement, and acceleration) of bridge structures modeled as 
multi-degree-of-freedom (MDOF) systems using a novel Deep Learning-Based Modal Combination (DC) rule. 
The DC rule enhances prediction accuracy by introducing modal contribution coefficients that account for both 
the characteristics of ground motion and the dynamic properties of the structural system. These coefficients are 
determined using a deep neural network (DNN) model.

In this framework, a single degree of freedom (DOF) is assigned to the longitudinal direction. 
The top four dominant modes are utilized to predict the displacement and acceleration of the bridge structures.

The software consists of two main components:
1. Developing the DNN model to predict the modal contribution coefficients.
2. Estimating structural responses by applying the DC rule.
For a comprehensive understanding of the theoretical foundation of the DC rule, please refer to the cited reference.

# Developers
Developed by Taeyong Kim (taeyongkim@ajou.ac.kr), Oh-Sung Kwon (os.kwon@utoronto.ca), and Junho Song (junhosong@snu.ac.kr)\
Ajou University, Seoul National University, and University of Toronto

# Reference
Kim, T., Kwon, O., and Song, J. (Accepted) Deep learning-based response spectrum analysis method for bridges 
subjected to bi-directional ground motions, Earthquake Engineering and Structural Dynamics, https://doi.org/10.1002/eqe.4345

# Required software and libraries
When constructing databases: Python 3.8 with Numpy version '1.24.3', Pandas version '2.0.3', Scipy version '1.10.1',OpenSeesPy
When developing DNN model: Python 3.10 with Numpy version '1.26.3', Pandas version '2.2', Scipy version '1.12.0', Tensorflow version '2.15.0'

# File descriptions
1. Construct_IM.py: This script estimates the intensity measure, which serves as input for the DNN model 
predicting modal contribution coefficients. The response spectrum size of 110 X 1 is employed as an intensity measure.
This code only provides three synthetmic ground motions for demonstration purposes. The response specturm values are saved in the
'Ground_info_bridge.npy' file.

2. Generate_bridges.py: This code is designed to generate different kinds of bridges. A total of 3-, 4- ,5- , 6- span bridges.
The structural characteristics of the bridge system is saved in the "generated_MDOF_systems" folder.

3. Construct_SDOF_database.py: This code is designed to calcualte the responses of each mode of structural systems.
The peak displacement and acceleration are caluclated and saved in the file. One needs OpenSees program. Plase put your OpenSees program 
in the same folder. Or you can specify the location of the OpenSees file in the "i_SDOF.py" script.

perform modal anlaysis. In other words, this code let you calculate the
peak responses of each mode. 

3. Construct_MDOF_database.py: This code is desinged to perform 



1. Construct_SDOF_database.py: This code is designed to construct a database containing the structural responses of various single-degree-of-freedom (SDOF) systems. 
It utilizes a 300-step period and 50-step damping coefficient. To execute this code successfully, 
it is necessary to download the NGA-WEST database ground motion acceleration. 
Note that only two artificially generated ground motions are provided within this code.

2. Construct_MDOF_structure.py: The purpose of this code is to generate diverse multi-degree-of-freedom (MDOF) systems. In this context, MDOF systems represent shear buildings with a degree of freedom for each story.

3. Construct_MDOF_database.py: This code is responsible for constructing a database of structural responses for various multi-degree-of-freedom (MDOF) systems. The seismic responses of SDOF systems are utilized for this purpose.

4. Construct_RS.py: This code estimates the response spectrum for each ground motion using the constructed SDOF database. The response spectrum is then used to calculate the modal responses in the "Construct_Data4DNN.py" script.


6. Construct_Data4DNN.py: This script preprocesses the dataset to prepare it as input for the DNN model.

7. Develop_DNN_model.py: This script outlines the process of training the DNN model for the DC rule. Hyperparameters should be updated in accordance with the dataset.

8. DCmodel.py: This script describes the prediction of seismic responses for structural systems using the DC rule. The DNN model is trained using the dataset in the reference. The results are then compared with those obtained using the SRSS rule.