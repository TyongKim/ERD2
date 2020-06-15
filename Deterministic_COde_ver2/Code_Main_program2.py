"""
This script is to predict structural responses (maximum transient displacement)


Created on Wed May 20 16:15:58 2020

@author: taeyongkim
"""


import numpy as np
from Choose_Hysteretic import Choose_Hysteretic

# Type hysteretic behaviors
Hys_character, Hys_index = Choose_Hysteretic()

# Import ground motions

