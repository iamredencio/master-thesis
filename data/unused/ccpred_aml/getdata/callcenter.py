from datetime import date, timedelta
import numpy as np
import pandas as pd

def get_ccd(df):
    ccd = dict()
    df = df.rename(columns={'row_date': 'datum'})
    ccd['E-support'] = df[['datum', 'Aanbod', 'ACD', 'Abandoned']].rename(
        columns={'Abandoned': 'Aban. Calls'})
    ccd['Bank & Leven'] = df[['datum', 'Aanbod (2)', 'ACD (2)', 'Abandoned (2)']].rename(
        columns={'Aanbod (2)': 'Aanbod', 'ACD (2)': 'ACD', 'Abandoned (2)': 'Aban. Calls'})
    ccd['Sparen & beleggen Totaal'] = df[['datum', 'Aanbod (3)', 'ACD (3)', 'Abandoned (3)']].rename(
        columns={'Aanbod (3)': 'Aanbod', 'ACD (3)': 'ACD', 'Abandoned (3)': 'Aban. Calls'})
    ccd['Hypotheken'] = df[['datum', 'Aanbod (4)', 'ACD (4)', 'Abandoned (4)']].rename(
        columns={'Aanbod (4)': 'Aanbod', 'ACD (4)': 'ACD', 'Abandoned (4)': 'Aban. Calls'})
    return ccd

def get_aanbod(dates, ccd):
    aanbod = list()
    
    for k in ccd.keys():
        df1 = ccd[k][['datum', 'Aanbod','Aban. Calls','ACD']]
        df2 = df1.set_index(pd.to_datetime(df1.datum)).drop('datum', axis=1)
        if pd.to_datetime(date.today()).dayofweek == 0:  # maandag >> vrijdag
            dmin1 = date.today() + timedelta(-3)
        else:
            dmin1 = date.today() + timedelta(-1)
        if df2['Aanbod'][dmin1] == 0:
            raise Exception('Aanbod op ' + dmin1.strftime('%d-%m-%Y') + ' voor team ' + k + ' niet beschikbaar.')
        df3 = pd.merge(left = df2[['Aanbod']].replace(0, np.nan).dropna(), right=df2[['Aban. Calls','ACD']], how='inner', left_index = True, right_index = True)
        left = pd.DataFrame(index=dates)
        df4 = pd.merge(left = left, right = df3, how='left', left_index = True, right_index = True)
        df4.ix[df4.index.weekday > 4, 'Aanbod'] = np.nan
        df5 = df4.fillna(method = 'ffill')
        aanbodname = 'Aanbod ' + k
        abanname = 'Aban. Calls ' + k
        acdname = 'ACD ' + k
        aanbod.append(df5.rename(columns={'Aanbod': aanbodname,
                                         'Aban. Calls': abanname,
                                         'ACD': acdname}))
    result = pd.concat(aanbod, axis=1)
    assert result.shape[0] == len(dates)
    return result

def get_prognose(dates, ccd):
    prognose = list()
    for k in ccd.keys():
        df1 = ccd[k][['datum', 'Forecast']].replace(0, np.nan).dropna()
        df2 = df1[['Forecast']].set_index(pd.to_datetime(df1.datum))
        left = pd.DataFrame(index=dates)
        df3 = pd.merge(left = left, right = df2, how='left', left_index = True, right_index = True)
        df3.ix[df3.index.weekday > 4, 'Forecast'] = np.nan
        df4 = df3.fillna(method = 'ffill')
        name = 'Forecast ' + k
        prognose.append(df4.rename(columns={'Forecast': name}))
    result = pd.concat(prognose, axis=1)
    assert result.shape[0] == len(dates)
    return result

def get_aanbod_vorig_jaar(dates, ccd):
    ma_window = 7
    aanbodvj = list()
    for k in ccd.keys():
        df1 = ccd[k][['datum', 'Aanbod']].replace(0, np.nan).dropna()
        df2 = df1[['Aanbod']].set_index(pd.to_datetime(df1.datum))
        maxahead = (dates.max() + timedelta(ma_window)).strftime("%Y%m%d")
        fulldates = pd.date_range(start='20140101', end = maxahead)
        left1 = pd.DataFrame(index=fulldates)
        df3 = pd.merge(left = left1, right = df2, how='left', left_index = True, right_index = True)
        df3.ix[df3.index.weekday > 4, 'Aanbod'] = np.nan
        df4 = df3.fillna(method = 'ffill')
        df4['Aanbod vorig jaar'] = pd.rolling_mean(df4.shift(365), window=ma_window, center=True)
        left2 = pd.DataFrame(index=dates)
        df5 = pd.merge(left=left2, right=df4[['Aanbod vorig jaar']], how='left', left_index=True, right_index=True)
        name = 'Aanbod vorig jaar ' + k
        aanbodvj.append(df5.rename(columns={'Aanbod vorig jaar': name}))
    result = pd.concat(aanbodvj, axis=1)
    assert result.shape[0] == len(dates)
    return result