import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.datasets import fetch_openml

dfs = {}

for file in Path("data").glob("*.csv"):
    dfs[(str(file).split(".")[0]).split("/")[-1]] = pd.read_csv(file)
    print((str(file).split(".")[0]).split("/")[-1] + "'s shape: " + str(dfs[(str(file).split(".")[0]).split("/")[-1]].shape))
