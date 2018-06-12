def get_unica(df, dates):
    df2 = df.set_index(pd.to_datetime(df['OUTPUTDATE'].astype('str'))).drop('OUTPUTDATE', axis=1)
    left = pd.DataFrame(index=dates)
    df3 = pd.merge(left=left, right=df2, how='left', left_index=True, right_index=True).fillna(method='ffill').fillna(
        method='bfill').replace(np.nan, 0)
    assert df3.shape[0] == len(dates)
    return df3