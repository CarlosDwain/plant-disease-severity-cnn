import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


def load_data(path_csv, sep=";"):
     dataset = pd.DataFrame(pd.read_csv(path_csv,sep=sep))
     print("\nShape dataset {}\n".format(dataset.shape))
     print(dataset.head())
     return dataset


def shuffle_split_data(path_csv, x_col, y_col, sep=";", validation=True, val_size=0.20, test_size=0.10):

    dataset = load_data(path_csv, sep)

    X = dataset[x_col]
    y = dataset[y_col]

    sss1 = StratifiedShuffleSplit(n_splits=1, random_state=0, test_size=test_size)
    for train_idx, test_idx in sss1.split(X, y):
        X_train, X_test =X.iloc[list(train_idx)], X.iloc[list(test_idx)]
        y_train, y_test =y.iloc[list(train_idx)], y.iloc[list(test_idx)]

    if validation:
        sss2 = StratifiedShuffleSplit(n_splits=1,  random_state=0, test_size=val_size)
        for train_idx, val_idx in sss2.split(X_train, y_train):
            X_train, X_val =X.iloc[list(train_idx)], X.iloc[list(val_idx)]
            y_train, y_val =y.iloc[list(train_idx)], y.iloc[list(val_idx)]

        print("Training set: {}\n".format(len(y_train)))
        print("Validation set: {}\n".format(len(y_val)))
        print("Test set: {}\n".format(len(y_test)))


        return X_train, y_train, X_val, y_val, X_test, y_test

    print("Training set: {}\n".format(len(y_train)))
    print("Test set: {}\n".format(len(y_test)))

    return X_train, y_train, X_test, y_test


def split_data(path_csv, x_col, y_col, sep=";", validation=True, val_size=0.20, test_size=0.10):

    dataset = load_data(path_csv, sep)

    X = dataset[x_col]
    y = dataset[y_col]

    sss1 = train_test_split(n_splits=1, random_state=0, test_size=test_size)
    for train_idx, test_idx in sss1.split(X, y):
        X_train, X_test =X.iloc[list(train_idx)], X.iloc[list(test_idx)]
        y_train, y_test =y.iloc[list(train_idx)], y.iloc[list(test_idx)]

    if validation:
        sss2 = train_test_split(n_splits=1,  random_state=0, test_size=val_size)
        for train_idx, val_idx in sss2.split(X_train, y_train):
            X_train, X_val =X.iloc[list(train_idx)], X.iloc[list(val_idx)]
            y_train, y_val =y.iloc[list(train_idx)], y.iloc[list(val_idx)]

        return X_train, y_train, X_val, y_val, X_test, y_test

    return X_train, y_train, X_test, y_test

def clean_dataframe(df, columns_to_check, df_name):
    """
    Clean a DataFrame by converting specified columns to numeric values, 
    removing rows with non-numeric values, and resetting the index.
    
    Args:
        df (pd.DataFrame): The DataFrame to be cleaned.
        columns_to_check (list): List of column names to check and convert to numeric.
    
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    for column in columns_to_check:
        # Convert the column to numeric (ignore errors for non-numeric values)
        df[column] = pd.to_numeric(df[column], errors='coerce')

        # Find rows with values other than 0 or 1
        invalid_rows = df[~df[column].isin([0, 1])]

        # Print the invalid rows for debugging
        if not invalid_rows.empty:
            print(f"Invalid rows in {df_name} in column '{column}':")
            print(invalid_rows)

        # Drop the invalid rows from the DataFrame
        df.drop(invalid_rows.index, inplace=True)

    # Reset the DataFrame index after removing rows
    df.reset_index(drop=True, inplace=True)

    return df
