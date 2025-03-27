# Software introduction
This software is designed to predict the responses—displacement and acceleration—of bridge structures modeled as multi-degree-of-freedom (MDOF) systems using a novel Deep Learning-Based Modal Combination (DC) rule. The DC rule enhances prediction accuracy by introducing modal contribution coefficients that account for both the characteristics of ground motion and the dynamic properties of the structural system. These coefficients are determined using a deep neural network (DNN) model.

In this framework, a single degree of freedom (DOF) is assigned in the longitudinal direction. The top four dominant modes are utilized to predict the displacement and acceleration of the bridge structures.

The software consists of two main components:

1. Developing the DNN model to predict the modal contribution coefficients.

2. Estimating structural responses by applying the DC rule.

For a comprehensive understanding of the theoretical foundation of the DC rule, please refer to the cited reference.

# Developers
Developed by:
Taeyong Kim (taeyongkim@ajou.ac.kr), Oh-Sung Kwon (os.kwon@utoronto.ca), and Junho Song (junhosong@snu.ac.kr)
Affiliations: Ajou University, Seoul National University, and the University of Toronto

# Reference
Kim, T., Kwon, O., and Song, J. (Accepted) Deep learning-based response spectrum analysis method for bridges 
subjected to bi-directional ground motions, Earthquake Engineering and Structural Dynamics, https://doi.org/10.1002/eqe.4345

# Required software and libraries
When constructing databases, the following software and libraries are required:

* Python 3.8.18

* NumPy 1.24.3

* OpenSeesPy 3.3.0

* PyTorch 2.3.0


# File descriptions
1. Construct_IM.py: This script estimates the intensity measure, which serves as an input for the DNN model predicting modal contribution coefficients. A 110 × 1 response spectrum is employed as the intensity measure. The script includes three synthetic ground motions for demonstration purposes. The response spectrum values are saved in the "Ground_info_bridge.npy" file.

2. Generate_bridges.py
This script generates different types of bridge structures, including 3-, 4-, 5-, and 6-span bridges. The structural characteristics of the generated bridge systems are saved in the "generated_MDOF_systems" folder.

3. Modal_analysis.py
This script estimates key modal properties of structural systems, which serve as inputs to the DNN model. The bridge structures generated using "Generate_bridges.py" are employed as target structures.

4. Construct_SDOF_database.py
This script calculates the response of each structural mode. The bridge structures generated through "Generate_bridges.py" are used. The peak displacement and acceleration values are computed and saved.

OpenSees is required to run this script.

Ensure that OpenSees is placed in the same directory, or specify its location in the "i_SDOF.py" script.

TCL scripts are automatically generated when running this script.

5. Construct_MDOF_database.py
This script calculates the seismic responses of bridge structures generated using "Generate_bridges.py". The OpenSeesPy library is employed for dynamic analysis.

Ensure compatibility between Python and OpenSeesPy, as version mismatches may cause errors.

For more information about OpenSeesPy, visit: OpenSeesPy Documentation.

6. Construct_Data4DNN.py
This script preprocesses the dataset to prepare it as an input for the DNN model.

To execute this script, the output data from "Modal_analysis.py," "Construct_SDOF_database.py," and "Construct_MDOF_database.py" is required.

7. Develop_DNN_model_displ.py
This script trains the DNN model to predict peak displacement using the DC rule.

Hyperparameters should be tuned according to the dataset.

PyTorch is used for model development.

8. Develop_DNN_model_accel.py
This script trains the DNN model to predict peak acceleration using the DC rule.

Hyperparameters should be tuned according to the dataset.

PyTorch is used for model development.

9. DCmodel_displ.py
This script predicts the peak displacement of structural systems using the DC rule.

The DNN model is trained using the dataset referenced in the publication.

Results obtained using the DC rule are compared with those derived from the Square Root of the Sum of Squares (SRSS) rule.

10. DCmodel_accel.py
This script predicts the peak acceleration of structural systems using the DC rule.

The DNN model is trained using the dataset referenced in the publication.

Results obtained using the DC rule are compared with those derived from the Square Root of the Sum of Squares (SRSS) rule.

