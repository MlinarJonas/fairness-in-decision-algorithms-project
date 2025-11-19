import kagglehub
import pandas as pd
import numpy as np
from tabulate import tabulate

# Download latest version
path = kagglehub.dataset_download("danofer/law-school-admissions-bar-passage")

# Load the dataset
data_file = path + "/" + "bar_pass_prediction.csv"
df = pd.read_csv(data_file, sep=",")
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)
print("Dataset loaded successfully.")

# shuffle dataset rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# drop rows with missing values
df = df.dropna()

# select features
features =['pass_bar', 'lsat','ugpa', 'gender', 'parttime', 'fulltime','American Indian','Asian American', 'Black', 'Mexican American', 'Other Hispanic', 'Puerto Rican', 'White']

# change to 0 and 1 for gender and fulltime
df['gender'] = df['gender'].map({'male': 0, 'female': 1})
df['fulltime'] = df['fulltime'].map({1 : 0, 2 : 1})

# One-hot encode race
races = ['American Indian','Asian American', 'Black', 'Mexican American', 'Other Hispanic', 'Puerto Rican', 'White', 'Other']
race_dummies = pd.get_dummies(df['race'], prefix='race', dummy_na=False)
race_dummies.columns = races
df = pd.concat([df, race_dummies], axis=1)

# Change to 0 and 1 for races
for race in races:
    df[race] = df[race].map({True: 1, False: 0})

# standarize ugpa, gpa and lsat
df['ugpa'] = (df['ugpa'] - df['ugpa'].mean()) / df['ugpa'].std()
df['gpa'] = (df['gpa'] - df['gpa'].mean()) / df['gpa'].std()
df['lsat'] = (df['lsat'] - df['lsat'].mean()) / df['lsat'].std()

# remove features that won't be used
df = df[features]

# function to return processed data
def get_data():
    return df

# function to print first n rows of dataset
def print_data():
    print(tabulate(df.head(), headers='keys', tablefmt='psql'))

# function to print dataset statistics
def print_stats():
    print(df.info())
    print(df.describe(include='all'))

# function to print dataset path
def print_path():
    print("Dataset path:", path)


