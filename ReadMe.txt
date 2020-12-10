This page provides the codes and the trained models for estimating hysteric models under random excitations. Each folder indicate different methods and summaries are as follows:


1. Deterministic_Code: The structural responses (Displacement, Velocity and Acceleration) under earthquake ground motions are estimated by using deep neural network. This codes are based on the paper: Kim, T., O.-S. Kwon*, and J. Song. (2019) Response Prediction of Nonlinear Hysteretic Systems by Deep Neural Networks. Neural Networks. Vol. 111, 1-10.

2. Probabilistic_Code: Extending the deterministic DNN model by using Bayesian framework. The Probabilistic DNN model produces the mean and variance of the structural responses. This codes are based on the paper: Kim, T., J. Song*, O.-S. Kwon. (2020) Probabilistic Evaluation of Seismic Responses Using Deep Learning Method, Structural Safety. Vol. 84, 101913.

<<<<<<< HEAD
3. Deterministic_Code_ver2: Recently, improved script of "1. Deterministic_Code" is written by the authors. The code provides three trained DNN model each of which can estimate maximum transient displacement, velocity and acceleration. Details can be found in README in the subfolder.
=======
3. Deterministic_Code_ver2: Recently, I improved script of "1. Deterministic_Code". The code provides three trained DNN models each of which can estimate maximum transient displacement, velocity and acceleration. Details can be found in README in the subfolder.
>>>>>>> 3f45f0b551f66be191455bfcfd11ab52c34e419f

This page will be updated whenever new methods are developed.
