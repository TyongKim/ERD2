# Software introduction
This software predicts the seismic responses (peak transient displacement) of structures having complex hysteretic behavior (stiffness/strength degradations and pinching effects). The software is comprised of (1) developing the deep neural network (DNN) model and (2) predicting structural responses using the DNN models. Details of the input information are described in each script. Finally, the code can be properly operated once the seismic database ('SDDB_mBWBN_ver1.0.db') is downloaded. The database is available at 'http://ERD2.snu.ac.kr'

# Developers
Developed by Taeyong Kim (taeyongkim@ajou.ac.kr), Oh-Sung Kwon (os.kwon@utoronto.ca), and Junho Song (junhosong@snu.ac.kr)\
Ajou University, Seoul National University, and University of Toronto

# Reference
Kim, T., Kwon, O., Song, J. (2023) Deep learning-based seismic response prediction of hysteretic systems having degradation and pinching. Earthquake Engineering and Structural Dynamics.
https://doi.org/10.1002/eqe.3796

# Required software and libraries
Python 3 with Numpy version '1.17.2', Pandas version '1.0.1', Scipy version '1.3.1', Keras version '2.3.0', Tensorflow version '1.14.0'

# File description (under construction)
1. Code_Make_DB.py: This script provides instructions on how to construct a seismic demand database used in this study. The script utilizes the m-BWBN model to generate multiple hysteretic behaviors, each with distinct structural characteristics. The database management system used is SQLite3. While two artificially simulated ground motions are included in this code due to copyright issues, the reference adopts 1,499 ground motions from the NGA-West database. Although the Runge-Kutta integration scheme is utilized to solve the equation of motion, other integration schemes can also be used.
2. Code_Generate_hysteresis.py: This script provides an introduction on how to generate hysteresis of the m-BWBN model in the form of an input for the ERD-net. A quasi-static cyclic analysis is performed using a backward Euler solution scheme provided in the Appendix of the reference. The generated hysteresis is then saved in the database and will be used to train the ERD-net.
3. Code_train_ERD_net.py: This script introduces how to train the ERD-net using the seismic demand database obtained from "Code_Make_DB.py" and the generated hysteresis from "Code_Generate_hysteresis.py". The preconstructed databases are available on ERD2 website (http://ERD2.snu.ac.kr).
4. Code_predict_displ.py: This script provides instructions on how to predict the seismic responses of a target structural system using the ERD-net.
5. ERD_net_displ.h5: Trained deep neural network (DNN) model, termed ERD-net, using 'Code_train_ERD_net.py'. The results of the above reference is obatined using this DNN model.




