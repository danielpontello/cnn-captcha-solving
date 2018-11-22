import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize

data = np.loadtxt("confusion_matrix_human.csv", delimiter=";")
conf_mat_norm = normalize(data, norm='l1')

np.savetxt("../results/confusion_matrix_homan_norm.csv", conf_mat_norm, delimiter = ";", fmt = "%.4f")