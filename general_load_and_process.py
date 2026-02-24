import os
from thz.dataset import DataSet, Constants
import numpy as np

fileDir = r"C:\Users\Samuel\Data\dispersion tests"
# fileDir = r"C:\Users\Samuel\Data\THz\Sam\2026-02-23_MINTS"

# files = [f for f in os.listdir(fileDir) if f.endswith('.csv') and f.startswith('tek')]

data_set = DataSet(fileDir)
data_set.load_all_data()
data_set.plot_current()