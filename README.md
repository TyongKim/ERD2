# ERD2 (ERD-base and ERD-net)

## Overview
This page provides the codes and the trained models for estimating hysteric models under random excitations. Each folder indicate different methods and summaries are as follows:

1. Deterministic_Code: The structural responses (Displacement, Velocity, and Acceleration) under earthquake ground motions are estimated using a deep neural network. The codes are based on the paper: Kim, T., O.-S. Kwon*, and J. Song. (2019) Response Prediction of Nonlinear Hysteretic Systems by Deep Neural Networks. Neural Networks. Vol. 111, 1-10.

2. Probabilistic_Code: 
The deterministic DNN model is extended by employing a Bayesian framework. The Probabilistic DNN model provides the mean and variance of the structural responses. The codes are based on the paper: Kim, T., J. Song*, O.-S. Kwon. (2020) Probabilistic Evaluation of Seismic Responses Using Deep Learning Method, Structural Safety. Vol. 84, 101913.

3. Deterministic_Code_ver2: Recently, the script for "1. Deterministic Code" has been improved. The code now includes three trained DNN models, each capable of estimating maximum transient displacement, velocity, and acceleration. Further details can be found in the README located in the subfolder.

4. m-BWBN_Code: A deep learning model has been developed to predict seismic responses (e.g., peak displacement) of hysteresis systems displaying stiffness/strength degradations and pinching effects. To achieve this, a modified Bouc-Wen-Baber-Noori model is proposed and utilized to construct a seismic demand database. The codes are based on the paper: Kim, T., Kwon, O., and Song, J. (2023). Deep learning-based seismic response prediction of hysteretic systems having degradation and pinching, Earthquake Engineering and Structural Dynamics, 52(8): 2384-2406.

5. MDOF_linear_building: A novel modal combination rule, named the Deep learning-based Combination (DC) rule, has been developed alongside the Square Root of the Sum of Squares (SRSS) and Complete Quadratic Combination (CQC). The DC rule introduces modal contribution coefficients predicted by a deep neural network model to consider different contributions of each modal response to the responses of multi-degree-of-freedom (MDOF) systems. The codes are based on the paper: Kim, T., Kwon, O., and Song, J. (2024). Deep learning-based response spectrum analysis method for building structures, Earthquake Engineering and Structural Dynamics.


## Developers
Developed by Taeyong Kim (taeyongkim@ajou.ac.kr), Oh-Sung Kwon (os.kwon@utoronto.ca), and Junho Song (junhosong@snu.ac.kr)


Institution: Ajou University, University of Toronto, and Seoul National University

## Reference
Kim, T., Kwon, O., Song, J. (2019) Response Prediction of Nonlinear Hysteretic Systems by Deep Neural Networks, Neural Networks. Neural Networks. Vol. 111, 1-10.

Kim, T., Song, J., Kwon, O. (2019) Probabilistic Estimation of Seismic Responses with Deep Neural Networks. Structural Safety Vol. 84, 101913.

Kim, T., Song, J., Kwon, O. (2020) Pre- and post-earthquake regional loss assessment using deep learning. Earthquake Engineering and Structural Dynamics.49: 657– 678.

Kim, T., Kwon, O., and Song, J. (2023). Deep learning-based seismic response prediction of hysteretic systems having degradation and pinching, Earthquake Engineering and Structural Dynamics, 52(8): 2384-2406. 

Kim, T., Kwon, O., and Song, J. (2024). Deep learning-based response spectrum analysis method for building structures, Earthquake Engineering and Structural Dynamics.

## Note
This page will be updated whenever new methods are developed.
