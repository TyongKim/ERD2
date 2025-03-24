# Software introduction
This software is designed to predict the responses of brdige structures represented by multi-degree-of-freedom (MDOF) 
systems utilizing a novel Deep Learning-based Modal Combination (DC) rule. 
The DC rule introduces modal contribution coefficients to improve prediction accuracy by accounting for 
both the characteristics of ground motion and the dynamic properties of a structural system. 
These coefficients are predicted by a deep neural network (DNN) model. 
We assing a single degree of freedom (DOF) to longitudinal direction. A total of top four modes are employed to predict the displacement
and acceleration of the 
The software consists of two main components: (1) developing the DNN model that predicts the modal contribution coefficients, 
and (2) predicting structural responses by applying the DC rule. For a more in-depth understanding of the theoretical foundation 
of the DC rule, please refer to the reference.

# Developers
Developed by Taeyong Kim (taeyongkim@ajou.ac.kr), Oh-Sung Kwon (os.kwon@utoronto.ca), and Junho Song (junhosong@snu.ac.kr)\
Ajou University, Seoul National University, and University of Toronto

# Reference
Kim, T., Kwon, O., and Song, J. (Accepted) Deep learning-based response spectrum analysis method for bridges subjected to bi-directional ground motions, Earthquake Engineering and Structural Dynamics, https://doi.org/10.1002/eqe.4345

# Required software and libraries
When constructing databases: Python 3.8 with Numpy version '1.24.3', Pandas version '2.0.3', Scipy version '1.10.1',OpenSeesPy
When developing DNN model: Python 3.10 with Numpy version '1.26.3', Pandas version '2.2', Scipy version '1.12.0', Tensorflow version '2.15.0'

# File description
1. Construct_SDOF_database.py: This code is designed to construct a database containing the structural responses of various single-degree-of-freedom (SDOF) systems. It utilizes a 300-step period and 50-step damping coefficient. To execute this code successfully, it is necessary to download the NGA-WEST database ground motion acceleration. Note that only two artificially generated ground motions are provided within this code.

2. Construct_MDOF_structure.py: The purpose of this code is to generate diverse multi-degree-of-freedom (MDOF) systems. In this context, MDOF systems represent shear buildings with a degree of freedom for each story.

3. Construct_MDOF_database.py: This code is responsible for constructing a database of structural responses for various multi-degree-of-freedom (MDOF) systems. The seismic responses of SDOF systems are utilized for this purpose.

4. Construct_RS.py: This code estimates the response spectrum for each ground motion using the constructed SDOF database. The response spectrum is then used to calculate the modal responses in the "Construct_Data4DNN.py" script.

5. Construct_IM.py: This script estimates the intensity measure, which serves as input for the DNN model predicting modal contribution coefficients.

6. Construct_Data4DNN.py: This script preprocesses the dataset to prepare it as input for the DNN model.

7. Develop_DNN_model.py: This script outlines the process of training the DNN model for the DC rule. Hyperparameters should be updated in accordance with the dataset.

8. DCmodel.py: This script describes the prediction of seismic responses for structural systems using the DC rule. The DNN model is trained using the dataset in the reference. The results are then compared with those obtained using the SRSS rule.