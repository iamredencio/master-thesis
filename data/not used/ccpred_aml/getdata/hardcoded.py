import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Dag van de week, dag van de maand als dummies

def calendar(dates):
    df_in = pd.DataFrame({'dayofweek': map(lambda date: date.dayofweek, dates),
                          'dayofmonth': map(lambda date: date.day, dates)}
                         , index=dates)
    enc = OneHotEncoder(dtype='int', sparse=False)
    df_out = pd.DataFrame(enc.fit_transform(df_in), index=df_in.index)
    assert df_out.shape[0] == len(dates)
    return df_out

# Aantal dagen na een vrije dag / vakantie

def days_after(dates, csv, name):
    df_in = pd.read_csv(csv)
    special_dates = pd.to_datetime(df_in.Datum)
    deltas = [(date, int((date - special).days)) for date in dates for special in special_dates if date >= special]
    df = pd.DataFrame.from_records(data=deltas, columns=['Datum', 'Delta'])
    mindeltas = df.groupby('Datum')['Delta'].min()
    df_out = pd.DataFrame({name: mindeltas},
                         index=dates)
    assert df_out.shape[0] == len(dates)
    return df_out

# Aantal dagen tot (belasting deadline)

def days_to(dates, csv, name):
    df_in = pd.read_csv(csv)
    special_dates = pd.to_datetime(df_in.Datum)
    deltas = [(date, int((special - date).days)) for date in dates for special in special_dates if date <= special]
    df = pd.DataFrame.from_records(data=deltas, columns=['Datum', 'Delta'])
    mindeltas = df.groupby('Datum')['Delta'].min()
    df_out = pd.DataFrame({name: mindeltas},
                         index=dates)
    assert df_out.shape[0] == len(dates)
    return df_out

