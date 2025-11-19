from sklearn.model_selection import train_test_split, KFold, StratifiedKFold


def train_val_test_split(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42, shuffle=True):
    train, temp = train_test_split(df, train_size=train_size, random_state=random_state, shuffle=shuffle)
    val, test = train_test_split(temp, test_size=test_size/(val_size+test_size), random_state=random_state, shuffle=shuffle)
    return train, val, test

def get_kfold(n_splits=5, shuffle=True, random_state=42):
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

def get_stratified_kfold(n_splits=5, shuffle=True, random_state=42):
    return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)



