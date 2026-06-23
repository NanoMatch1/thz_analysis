# load the pickled dataset and plot the derived conductivity (sigma) for each sample in the dataset.

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
scriptDir = os.path.dirname(os.path.abspath(__file__))
pickle_path = [file for file in os.listdir(scriptDir) if file.endswith(".pkl")][0]

with open(os.path.join(scriptDir, pickle_path), "rb") as f:
    dataset = pickle.load(f)

print(f"Loaded dataset from {pickle_path} with {len(dataset)} items.")
breakpoint()

