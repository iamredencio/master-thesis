import numpy as np
import pandas as pd


def prefix_column(df, prefix):
    """Add a prefix to the new k_ and t_ attributes"""
    newnames = map(lambda oldname: prefix + oldname, df.columns)
    df.columns = newnames
    return df

def k_attributes(df, ahead = 1):
    # ahead hoeveel dagen terug moeten de attributes komen, hoeveel dagen wil je vooruit voorspellen?
    # attributes krijgen toevoeging K van Known (meest recent bekende info, gezien de forecast)

    df1 = df.shift(ahead)
    prefix_column(df1, 'K1_')
    df2 = df.shift(ahead + 1)
    prefix_column(df2, 'K2_')
    df3 = df.shift(ahead + 2)
    prefix_column(df3, 'K3_')
    df4 = df.shift(ahead + 3)
    prefix_column(df4, 'K4_')
    df5 = df.shift(ahead + 5)
    prefix_column(df5, 'K5_')

    df6 = pd.rolling_mean(df.shift(ahead), window=7, center=False)
    prefix_column(df6, 'K1_7_')
    df7 = pd.rolling_mean(df.shift(ahead), window=14, center=False)
    prefix_column(df7, 'K1_14_')
    df8 = pd.rolling_mean(df.shift(ahead), window=21, center=False)
    prefix_column(df8, 'K1_21_')
    df9 = pd.rolling_mean(df.shift(ahead), window=28, center=False)
    prefix_column(df9, 'K1_28_')

    result = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9], axis=1)
    return result


def t_attributes(df):
    # ahead hoeveel dagen terug moeten de attributes komen, hoeveel dagen wil je vooruit voorspellen?
    # attributes krijgen toevoeging K van Known (meest recent bekende info, gezien de forecast)

    base = pd.Series(np.arange(df.shape[0]), index=df.index)

    window = 2
    df1 = df.apply(lambda c: pd.rolling_cov(base, c, window=window) / pd.rolling_var(base, window=window, center=False), axis=0)
    prefix_column(df1, 'T1_2_')
    window = 3
    df2 = df.apply(lambda c: pd.rolling_cov(base, c, window=window) / pd.rolling_var(base, window=window, center=False), axis=0)
    prefix_column(df2, 'T1_3_')
    window = 5
    df3 = df.apply(lambda c: pd.rolling_cov(base, c, window=window) / pd.rolling_var(base, window=window, center=False), axis=0)
    prefix_column(df3, 'T1_5_')
    window = 7
    df4 = df.apply(lambda c: pd.rolling_cov(base, c, window=window) / pd.rolling_var(base, window=window, center=False), axis=0)
    prefix_column(df4, 'T1_7_')
    window = 14
    df5 = df.apply(lambda c: pd.rolling_cov(base, c, window=window) / pd.rolling_var(base, window=window, center=False), axis=0)
    prefix_column(df5, 'T1_14_')
    window = 21
    df6 = df.apply(lambda c: pd.rolling_cov(base, c, window=window) / pd.rolling_var(base, window=window, center=False), axis=0)
    prefix_column(df6, 'T1_21_')
    window = 28
    df7 = df.apply(lambda c: pd.rolling_cov(base, c, window=window) / pd.rolling_var(base, window=window, center=False), axis=0)
    prefix_column(df7, 'T1_28_')

    result = pd.concat([df1, df2, df3, df4, df5, df6, df7], axis=1)
    return result
