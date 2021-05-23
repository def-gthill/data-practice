"""
Data preparation and cleaning functions
"""

import pandas as pd


def split_y(df, y_column):
    """
    Splits the y column out of a dataframe containing
    both x and y columns.
    
    Returns a dataframe containing the x columns
    and a dataframe containing only the y column
    """
    return df.drop(y_column, axis=1), df[[y_column]]


class ColumnAssigner:
    """
    A scikit-learn transformer that converts a numpy
    array into a Pandas dataframe with the specified
    column names. Useful for pipelines where later
    steps need column names (e.g. ColumnTransformer)
    but earlier steps strip the column names off
    (e.g. every single built-in transformer).
    """
    
    def __init__(self, column_names):
        self.column_names = column_names
    
    def fit(self, data, targets=None):
        return self
    
    def transform(self, data):
        return pd.DataFrame(data, columns=self.column_names)


class ColumnKeeper:
    """
    A scikit-learn transformer that keeps only the columns
    of the dataframe with the specified names.
    """
    
    def __init__(self, column_names):
        self.column_names = column_names
    
    def fit(self, data, targets=None):
        return self
    
    def transform(self, data):
        return data[self.column_names]


class ColumnDropper:
    """
    A scikit-learn transformer that drops the columns
    of the dataframe with the specified names.
    
    If the data passed to transform has different
    column names than the data passed to fit,
    the transformed dataframe will have the columns
    that the fit dataframe would have had if the
    columns had been dropped.
    
    For example, if a dataframe with the columns
    ['foo', 'bar', 'baz'] is passed to fit,
    and columns is ['bar'], then transform
    will always keep the 'foo' and 'baz' columns
    and only those columns, regardless of what
    the rest of the dataframe looks like.
    
    That way, if there are irrelevant columns
    in the training data and different irrelevant
    columns in the production data, the column
    dropper will keep the same set of columns
    from each.
    """
    
    def __init__(self, column_names):
        self.columns_to_drop = column_names
    
    def fit(self, data, targets=None):
        columns_to_drop_set = set(self.columns_to_drop)
        self.columns_to_keep = [
            column for column in data.columns
            if column not in columns_to_drop_set
        ]
        return self
    
    def transform(self, data):
        return data[self.columns_to_keep]

