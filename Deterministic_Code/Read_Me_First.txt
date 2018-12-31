The codes show how to evaluate the responses of a nonlinear hysteretic system under a random vibration using a deep learning method. In this example, the nonlinear hysteric system is the structure represented by a single degree of freedom (SDOF) system, while the random vibration is an earthquake ground motion. The codes are based on the paper "Kim T, Kwon OS, and Song J, 2019" published at Neural Networks. The paper’s full reference information is as follows:

Kim, T., O.-S. Kwon*, and J. Song. (2019) Response Prediction of Nonlinear Hysteretic Systems by Deep Neural Networks. Neural Networks.

The example codes are developed based on Python with Keras and Tensorflow. Moreover, the user interface for estimating structural responses is provided on http://erd2.snu.ac.kr and the database (DB) can be downloaded on the site (Because the size of the DB is too large that we cannot upload on Github). Once you get the database, you can use the Python codes more properly (The URL will be soon available).

The description of the codes are as follows:
1. Generate_model.py: This code is for generating the deep neural network (DNN) model for estimating the nonlinear hysteric system under the random excitation. This is the raw model so that one can freely modify the architecture or training method to get his/her own model.

2. Prediction.py: This code is an example of how to use the trained DNN model to estimate the structural responses (Displacement, Velocity, and Acceleration). The trained models are saved for h5 file.

Remark:
- We also provide the hysteresis of the SDOF system for linear, bilinear, and bilinear with hysteretic degradation (Total 54,090). Details are provided in the paper. One could freely modify the architecture of the deep neural network and use the trained models.

