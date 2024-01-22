# Software introduction
This software is designed to predict the responses of multi-degree-of-freedom (MDOF) systems utilizing a novel Deep Learning-based Modal Combination (DC) rule. The DC rule introduces modal contribution coefficients to improve prediction accuracy by accounting for both the characteristics of ground motion and the dynamic properties of a structural system. These coefficients are predicted by a deep neural network (DNN) model. The software consists of two main components: (1) developing the DNN model that predicts the modal contribution coefficients, and (2) predicting structural responses by applying the DC rule. For a more in-depth understanding of the theoretical foundation of the DC rule, please refer to the reference.

# Developers
Developed by Taeyong Kim (taeyongkim@ajou.ac.kr), Oh-Sung Kwon (os.kwon@utoronto.ca), and Junho Song (junhosong@snu.ac.kr)\
Ajou University, Seoul National University, and University of Toronto

# Reference
Kim, T., Kwon, O., and Song, J. (Accepted). Deep learning-based response spectrum analysis method for building structures, Earthquake Engineering and Structural Dynamics. https://doi.org/10.1002/eqe.4086

# Required software and libraries
Python 3 with Numpy version '1.24.3', Pandas version '1.5.3', Scipy version '1.10.1', Tensorflow version '2.14.0'

# File description
1. Construct_SDOF_database.py: This is the code for constructing a database of structural responses of various single-degree-of-freedom (SDOF) systems. 300 steps period and 50 steps damping coefficient. In order to run this code, you may want to download the NGA-WEST database ground motion acceleration. Only two artificially generated ground motions are provided in this code.\

2. Construct_MDOF_structure.py: This code aims to generate various multi-degree-of-freedom (MDOF) systems. In here, MDOF systems represent shear buildingsconsisting degree of freedom for each story.\

3. Construct_MDOF_database.py: This is the code for constructing a database of structural responses of various multi-degree-of-freedom (MDOF) systems. The seismic responses of SDOF systems are employed to this end.\

4. Construct_RS.py: This code esitmates the response specturm for each ground motion using the constructed SDOF database. The response spectrum is used to calcualte the modal responses in the script of "Construct_Data4DNN.py."\

5. Construct_IM.py: The script estimates the intensity measure which is used as inputs for the DNN model predicting modal contribution coefficients.\

6. Construct_Data4DNN.py: The script preprocess dataset to make them as inputs for the DNN model.\

7. Develop_DNN_model.py: 


