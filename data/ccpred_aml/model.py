
from datetime import date, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from ccpred_aml import config
from ccpred_aml import merge
from ccpred_aml.getdata import callcenter

def forecast(ccd_df, unica_df, team=config.team, datadir=config.datadir, window=config.window, n_features=100, maxahead=config.maxahead):
    """Produces the forecasts of the number of calls.

    Parameters
    ----------
    y : string
        The call center team for which the number of calls are forecasted

    datadir : string
        Directory where csv's about holidays and tax dates are stored

    window : integer
        The training window, the number of days used to train the model

    n_features : integer
        The number of most important features that are passed to the random forest and elastic net.
        Selection is bases on a first random forest that includes all features
    """
    ccd = callcenter.get_ccd(ccd_df)
    apda = merge.get_apda(config.dates, ccd, unica_df)
    akda = merge.get_akda(config.dates, datadir, ccd)
    allX = merge.get_allX(akda, apda, datadir, maxahead)
    y = merge.get_y(config.dates, ccd, team, datadir)
    recs = list()
    rfs = dict()
    for k in sorted(allX.keys()):
        # print 'k:\t', k
        ahead = int(k[1:])
        # print 'ahead:\t', ahead
        forecast_day = pd.to_datetime(date.today() + timedelta(ahead-1))
        # print 'forecast_day:\t', forecast_day
        X_forecast = allX[k][allX[k].index == forecast_day]
        # print 'X_forecast:'
        # display(X_forecast)
        if X_forecast.shape[0] == 0:
            y_forecast = np.nan
        else:
            X_train_prem = allX[k][allX[k].index < forecast_day]
            y_train_prem = y[y.index < forecast_day]
            X_train = X_train_prem[X_train_prem.index.isin(y_train_prem.index)][-window:]
            y_train = y_train_prem.iloc[y_train_prem.index.isin(X_train_prem.index), 0][-window:]
            # print 'X_train.shape', X_train.shape, min(X_train.index.date), max(X_train.index.date), 'y_train.shape', y_train.shape, min(y_train.index.date), max(y_train.index.date)
            rf1 = RandomForestRegressor(n_estimators=100, criterion='mse', max_features='sqrt', random_state=42)
            rf1.fit(X_train, y_train)
            fi = pd.DataFrame({'Importance': rf1.feature_importances_
                                  , 'Feature': X_train.columns.values})
            fi.sort(columns='Importance', ascending=False, inplace=True)
            X_train2 = X_train[fi.Feature[:n_features]]
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train2, y_train)
            rfs[k] = [rf1,rf]
            X_forecast2 = X_forecast[fi.Feature[:n_features]]
            y_forecast = int(round(rf.predict(X_forecast2)[0]))
        # print y_forecast
        recs.append((forecast_day.date(), ahead, y_forecast))
    # print recs
    df = pd.DataFrame.from_records(data=recs, index='forecast_day', columns=['forecast_day', 'ahead', 'forecast'])
    return df
