import pandas as pd
import numpy as np

def ms_2_bf(ms):
    if(0 <= ms <= 2):
        bf = 0
    elif(3 <= ms <= 15):
        bf = 1
    elif(16 <= ms <= 33):
        bf = 2
    elif(34 <= ms <= 54):
        bf = 3
    elif(55 <= ms <= 79):
        bf = 4
    elif(80 <= ms <= 107):
        bf = 5
    elif(108 <= ms <= 138):
        bf = 6
    elif(139 <= ms <= 171):
        bf = 7
    elif(172 <= ms <= 207):
        bf = 8
    elif(208 <= ms <= 244):
        bf = 9
    elif(245 <= ms <= 284):
        bf = 10
    elif(285 <= ms <= 326):
        bf = 11
    else:
        bf = 12
    return(bf)

def storm(bf):
    if bf >= 9:
        storm = 1
    else:
        storm = 0
    return(storm)

def get_knmi(df, dates):
    df2 = df.replace(to_replace='', value=np.nan).fillna(value=np.nan)
    df3 = df2.apply(lambda c: c.astype(np.float64), axis=1)
    df4 = df3.set_index(pd.to_datetime(df['YYYYMMDD'].astype('str'))).drop('YYYYMMDD', axis=1)
    df4['BF'] = map(lambda x: ms_2_bf(x), df4['FXX'])
    df4['STORM'] = map(lambda x: storm(x), df4['BF'])
    left = pd.DataFrame(index=dates)
    df5 = pd.merge(left = left, right = df4, how='left', left_index = True, right_index = True)
    df6 = df5.fillna(method='ffill')
    assert df6.shape[0] == len(dates)
    return df6

