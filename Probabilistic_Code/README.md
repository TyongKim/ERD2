# Software introduction
This software predicts structural responses (peak displacement, velocity, and acceleration) using deep neural networks in a probabilistic manner. The software is comprised of (1) developing the probabilistic deep neural network (P-DNN) models and (2) predicting structural responses using the P-DNN models. Details of the input information are described in each script. Finally, the code can be properly operated once the seismic database ('SDDB_v1.0.2.db') is downloaded. The database can be downloaded at 'http://ERD2.snu.ac.kr'

# Developers
Developed by Taeyong Kim (chs5566@snu.ac.kr), Oh-Sung Kwon (os.kwon@utoronto.ca), and Junho Song (junhosong@snu.ac.kr)\
Seoul National University and University of Toronto

# Reference
Kim, T., Song, J., and Kwon, O.S. (2020). Probabilistic evaluation of seismic responses using deep learning method. Structural Safety. Vol. 84, 101913.
https://doi.org/10.1016/j.strusafe.2019.101913

# Required software and libraries
Python 3 with Numpy version '1.17.2', Pandas version '1.0.1', Scipy version '1.3.1', Keras version '2.3.0', Tensorflow version '1.14.0'

# File description
1. Code_Generate_PDNN_model.py: This script describes how we can train the P-DNN model to predict structural responses given different single degree of freedom system (SDOF) with various ground motions. Thanks to Keras, it is possible to find out the architecture of P-DNN model at glance. The code is developed to predict the peak displacement of structural systems, but the P-DNN model for other structural responses can be readily developed by just chaining a few lines in the code.\
2. Code_Main_run.py: This script probabilistically estimates structural responses based on the user-defined input in 'Code_Main_script.py'.\
3. Code_Main_script.py: This script let the users define target structural systems and earthquake information of interest. After defining the information, the responses can be provided by running 'Code_Main_run.py'.\
4. Code_Predict_seismic_responses.py: This script is a general version of 'Code_Main_run.py' and 'Code_Main_script.py', which can employ various structural systems and earthquake motions together. The code is developed to help the users apply the P-DNN model to their research. This code provides the same results that can be obtained from 'Code_Main_run.py'.\
5. P_DNN_model_2019_accel.h5, P_DNN_model_2019_displ.h5, P_DNN_model_2019_velo.h5: Trained deep neural network (DNN) models using 'Code_Generate_DNN_model.py'.\
6. EX_1970_Wrightwood_6074_Park_Dr_North.AT2: Accelerogram of 1970 Wrightwood 6074 Park Dr North earthquake downloaded at NGA-West2 Database. This is used as an example of how we use the P-DNN model. Please note that this accelerogram is not used in training nor included in the seismic demand database 'SDDB_v1.0.2.db'.\
7. EX_GM0.txt: Converted accelerogram of at2 file to txt file.\
8. EX_RS_1970_Wrightwood.csv: Response spectrum data of the example earthquake obtained using the Newmark-beta method.\
9. EX_time0.txt: Time step of the example accelerogram which can be obtained in '.AT2' file.\
10. Newmark_TK.py: Newmark method that calculates the responses of a linear SDOF system. The code is employed when estimating structural responses once accelerograms are provided.\
11. Period.npy: Period steps for the response spectrum of the DNN input. I use the response spectrum of these period steps when training the DNN model. Thus, one should follow this step for properly estimating structural responses.\
12. plot_hysteretic_data.npy: The results of quasi-cyclic analysis (force and displacement steps) of a various SDOF system. This hysteresis has been employed as a structural input of the DNN model. Note that the index of the dataset corresponds to the 'structural_index.npy'.\ 
13. structural_index.npy: The characteristics of SDOF system. Period, stiffness, yield force and post-yield stiffness ratio are provided. We consider 90 linear system (HM1), 27,000 bilinear system (HM2), and 27,000 bilinear with stiffness degradation system (HM3). Thus, a total of 54,090 systems are provided.\