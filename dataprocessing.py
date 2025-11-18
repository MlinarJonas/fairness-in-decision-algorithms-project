import kagglehub
import pandas as pd
import numpy as np
from tabulate import tabulate

# Download latest version
path = kagglehub.dataset_download("danofer/law-school-admissions-bar-passage")
print("Path to dataset files:", path)

# Load the dataset
data_file = path + "/" + "bar_pass_prediction.csv"
df = pd.read_csv(data_file, sep=",")
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)
print("Dataset loaded successfully.")

print(df.info())
print(df.describe(include='all'))
print(tabulate(df.iloc[:, :8].head(),headers='keys', tablefmt='psql'))





