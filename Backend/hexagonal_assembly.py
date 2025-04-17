import pickle
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os 

# Read in the data
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"data.pkl"), "rb") as f:
    data = pickle.load(f)
    
    # Inject into global namespace
    globals().update(data)

print(sigma_ss_u38)