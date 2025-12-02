import kagglehub
import pandas as pd
from tabulate import tabulate
from data.data_split import train_val_test_split

download_path = "danofer/law-school-admissions-bar-passage"
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)

def download_dataset(download_path = download_path):
    path = kagglehub.dataset_download(download_path)
    return path

def load_dataset(path, separator = ","):
    data_file = path + "/" + "bar_pass_prediction.csv"
    df = pd.read_csv(data_file, sep=separator)
    return df

def process_dataset(df: pd.DataFrame, dropna: bool=True, standardize: bool=True, shuffle: bool=True):
    df = df.copy()
    if shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True) # shuffle dataset rows
    if dropna:
        df = df.dropna() # drop rows with missing values
    
    # select features
    features = ['pass_bar', 'lsat','ugpa', 'gender', 'parttime', 'fulltime','American Indian','Asian American', 'Black', 'Mexican American', 'Other Hispanic', 'Puerto Rican', 'White']

    # change to 0 and 1 for gender and fulltime
    df['gender'] = df['gender'].map({'male': 0, 'female': 1})
    df['fulltime'] = df['fulltime'].map({1 : 0, 2 : 1})

    # One-hot encode race
    races = ['American Indian','Asian American', 'Black', 'Mexican American', 'Other Hispanic', 'Puerto Rican', 'White', 'Other']
    race_dummies = pd.get_dummies(df["race"])
    race_dummies = race_dummies.reindex(columns=races, fill_value=0)
    df = pd.concat([df, race_dummies], axis=1)

    # standarize ugpa, gpa and lsat
    if standardize:
        df['ugpa'] = (df['ugpa'] - df['ugpa'].mean()) / df['ugpa'].std()
        df['gpa'] = (df['gpa'] - df['gpa'].mean()) / df['gpa'].std()
        df['lsat'] = (df['lsat'] - df['lsat'].mean()) / df['lsat'].std()

    # remove features that won't be used
    df = df[features]

    return df

def print_data(df, rows=5):
    print(tabulate(df.head(rows), headers='keys', tablefmt='psql'))

def print_dataset_stats(df):
    print(df.info())
    print(df.describe(include='all'))

def prepare_data_for_training(df):
    train, val, test = train_val_test_split(df)

    # Remove pass_bar from features for training set
    X_train = train.drop(columns=['pass_bar']).to_numpy()
    y_train = train['pass_bar'].to_numpy()

    # remove pass_bar from features for validation set
    X_val = val.drop(columns=['pass_bar']).to_numpy()
    y_val = val['pass_bar'].to_numpy()
    gender_val = val['gender'].to_numpy()

    return (X_train, y_train), (X_val, y_val, gender_val), test

def get_feature_names(df):
    feature_names = df.drop(columns=['pass_bar']).columns.tolist()
    return feature_names