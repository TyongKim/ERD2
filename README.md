# ERD2 (ERD-base and ERD-net)

## Overview
This page provides the codes and the trained models for estimating hysteric models under random excitations. Each folder indicate different methods and summaries are as follows:

1. Deterministic_Code: The structural responses (Displacement, Velocity and Acceleration) under earthquake ground motions are estimated by using deep neural network. The codes are based on the paper: Kim, T., O.-S. Kwon*, and J. Song. (2019) Response Prediction of Nonlinear Hysteretic Systems by Deep Neural Networks. Neural Networks. Vol. 111, 1-10.

2. Probabilistic_Code: Extending the deterministic DNN model by using Bayesian framework. The Probabilistic DNN model produces the mean and variance of the structural responses. The codes are based on the paper: Kim, T., J. Song*, O.-S. Kwon. (2020) Probabilistic Evaluation of Seismic Responses Using Deep Learning Method, Structural Safety. Vol. 84, 101913.

3. Deterministic_Code_ver2: Recently, I improved script of "1. Deterministic_Code". The code provides three trained DNN models each of which can estimate maximum transient displacement, velocity and acceleration. Details can be found in README in the subfolder.

4. m-BWBN_Code: A deep learning model is developed to predict the seismic responses (e.g., peak displacement) of a hysteresis showing stiffness/strength degradations and pinching effects. To this end, a modified Bouc-Wen-Baber-Noori model is proposed and used to construct a seismic demand database. The codes are based on the paper: Kim, T., Kwon, O., Song, J. (2023) Deep learning-based seismic response prediction of hysteretic systems having degradation and pinching. Earthquake Engineering and Structural Dynamics.


## Developers
Developed by Taeyong Kim (taeyongkim@ajou.ac.kr), Oh-Sung Kwon (os.kwon@utoronto.ca), and Junho Song (junhosong@snu.ac.kr)


Institution: Ajou University, University of Toronto, and Seoul National University

## Reference
Kim, T., Kwon, O., Song, J. (2019) Response Prediction of Nonlinear Hysteretic Systems by Deep Neural Networks, Neural Networks. Neural Networks. Vol. 111, 1-10.

Kim, T., Song, J., Kwon, O. (2019) Probabilistic Estimation of Seismic Responses with Deep Neural Networks. Structural Safety Vol. 84, 101913.

Kim, T., Song, J., Kwon, O. (2020) Pre- and post-earthquake regional loss assessment using deep learning. Earthquake Engineering and Structural Dynamics.49: 657– 678.

Kim, T., Kwon, O., Song, J. (2023) Deep learning-based seismic response prediction of hysteretic systems having degradation and pinching. Earthquake Engineering and Structural Dynamics.

## Note
This page will be updated whenever new methods are developed.
