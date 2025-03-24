# ERD2 (ERD-base and ERD-net)

## Overview
This page provides the codes and the trained models for estimating hysteric models under random excitations. Each folder indicate different methods and summaries are as follows:

1. Deterministic_Code: This script estimates structural responses (displacement, velocity, and acceleration) under earthquake ground motions using a deep neural network (DNN). The implementation is based on the study by Kim, T., O.-S. Kwon*, and J. Song (2019), Response Prediction of Nonlinear Hysteretic Systems by Deep Neural Networks, Neural Networks, Vol. 111, pp. 1–10.

2. Probabilistic_Code: This script extends the deterministic DNN model by incorporating a Bayesian framework. The probabilistic DNN model provides both the mean and variance of structural responses. The implementation is based on the study by Kim, T., J. Song*, and O.-S. Kwon (2020), Probabilistic Evaluation of Seismic Responses Using a Deep Learning Method, Structural Safety, Vol. 84, Article 101913.

3. Deterministic_Code_ver2: This updated version of the Deterministic_Code script now includes three trained DNN models, each capable of estimating maximum transient displacement, velocity, and acceleration. Additional details are available in the README file located in the corresponding subfolder.

4. m-BWBN_Code: A deep learning model has been developed to predict the seismic responses (e.g., peak displacement) of hysteretic systems exhibiting stiffness/strength degradation and pinching effects. To facilitate this, a modified Bouc-Wen-Baber-Noori (m-BWBN) model has been proposed and employed to construct a seismic demand database. The implementation is based on the study by Kim, T., O. Kwon, and J. Song (2023), Deep Learning-Based Seismic Response Prediction of Hysteretic Systems with Degradation and Pinching, Earthquake Engineering and Structural Dynamics, 52(8), pp. 2384–2406.

5. MDOF_linear_building: A novel modal combination rule, termed the Deep Learning-Based Combination (DC) rule, has been developed alongside the Square Root of the Sum of Squares (SRSS) and Complete Quadratic Combination (CQC) methods. The DC rule introduces modal contribution coefficients predicted by a deep neural network model, accounting for the varying contributions of individual modal responses to the overall response of multi-degree-of-freedom (MDOF) systems. The implementation is based on the study by Kim, T., O. Kwon, and J. Song (2024), Deep Learning-Based Response Spectrum Analysis Method for Building Structures, Earthquake Engineering and Structural Dynamics.

6. MDOF_linear_bridge: The Deep Learning-Based Combination (DC) rule has also been developed for bridge structures subjected to bi-directional ground motions. The implementation is based on the study by Kim, T., O. Kwon, and J. Song (2025), Deep Learning-Based Response Spectrum Analysis Method for Bridges Subjected to Bi-Directional Ground Motions, Earthquake Engineering and Structural Dynamics.


## Developers
Developed by Taeyong Kim (taeyongkim@ajou.ac.kr), Oh-Sung Kwon (os.kwon@utoronto.ca), and Junho Song (junhosong@snu.ac.kr)


Institution: Ajou University, University of Toronto, and Seoul National University

## Reference
Kim, T., Kwon, O., Song, J. (2019) Response Prediction of Nonlinear Hysteretic Systems by Deep Neural Networks, Neural Networks. Neural Networks. Vol. 111, 1-10.

Kim, T., Song, J., Kwon, O. (2019) Probabilistic Estimation of Seismic Responses with Deep Neural Networks. Structural Safety Vol. 84, 101913.

Kim, T., Song, J., Kwon, O. (2020) Pre- and post-earthquake regional loss assessment using deep learning. Earthquake Engineering and Structural Dynamics.49: 657– 678.

Kim, T., Kwon, O., and Song, J. (2023). Deep learning-based seismic response prediction of hysteretic systems having degradation and pinching, Earthquake Engineering and Structural Dynamics, 52(8): 2384-2406. 

Kim, T., Kwon, O. S., & Song, J. (2024). Deep learning‐based response spectrum analysis method for building structures. Earthquake Engineering & Structural Dynamics, 53(4), 1638-1655.

Kim, T., Kwon, O., and Song, J. (2025). Deep learning-based response spectrum analysis method for bridges subjected to bi-directional ground motions, Earthquake Engineering and Structural Dynamics.

## Note
This page will be updated whenever new methods are developed.
