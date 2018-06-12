# Get all day features that are known beforehand
import os
import numpy as np
import pandas as pd
from ccpred_aml import featurize
from ccpred_aml.getdata import hardcoded, callcenter
from datetime import date, timedelta


# Get all day features that are known beforehand

def get_akda(dates, datadir, ccd):
    calendar_df = hardcoded.calendar(dates)
    vrij_df = hardcoded.days_after(dates, os.path.join(datadir,'vrije_dagen.csv'), 'dagen_na_vrij')
    noord_df = hardcoded.days_after(dates, os.path.join(datadir,'noord_vakanties.csv'), 'dagen_na_vnoord')
    midden_df = hardcoded.days_after(dates, os.path.join(datadir,'midden_vakanties.csv'), 'dagen_na_vmidden')
    zuid_df = hardcoded.days_after(dates, os.path.join(datadir,'zuid_vakanties.csv'), 'dagen_na_vzuid')
    belasting_df = hardcoded.days_to(dates, os.path.join(datadir,'belasting_dagen.csv'), 'dagen_tot_belasting')
    avj_df = callcenter.get_aanbod_vorig_jaar(dates, ccd)
    result = pd.concat([calendar_df, vrij_df, noord_df, midden_df, zuid_df, belasting_df, avj_df], axis=1)
    assert result.shape[0] == len(dates)
    return result

# Get all day features that are not known beforehand

# def get_apda(dates, ccd):
def get_apda(dates, ccd, unica_df):
    aanbod_df = callcenter.get_aanbod(dates, ccd)
    # result = pd.concat([aanbod_df, unica_df, knmi_df], axis=1)
    result = pd.concat([aanbod_df, unica_df], axis=1)
    assert result.shape[0] == len(dates)
    return result


def get_sapda(dates, ccd, sccd, MKey):
    aanbod_df = callcenter.get_aanbod(dates, ccd)
    saanbod_df = callcenter.get_saanbod(dates, sccd)
    unica_df = oracle.get_unica(dates)
    # voorraad_df = oracle.get_werkvoorraden(dates)
    svoorraad_df = hardcoded.get_swerkvoorraden(dates)
    # knmi_df = knmi.get_data(open('H:/knmi_de_bilt.txt', 'rb').read(), dates)
    knmi_df = knmi.get_knmi(knmi.get_tekst(MKey), dates)
    # return pd.concat([aanbod_df, saanbod_df, unica_df, knmi_df], axis=1)
    result = pd.concat([aanbod_df, saanbod_df, unica_df, knmi_df, svoorraad_df], axis=1)
    assert result.shape[0] == len(dates)
    return result


def get_papda(dates, ccd, pccd, MKey):
    aanbod_df = callcenter.get_aanbod(dates, ccd)
    paanbod_df = callcenter.get_paanbod(dates, pccd)
    unica_df = oracle.get_unica(dates)
    knmi_df = knmi.get_knmi(knmi.get_tekst(MKey), dates)
    result = pd.concat([aanbod_df, paanbod_df, unica_df, knmi_df], axis=1)
    assert result.shape[0] == len(dates)
    return result


# Merge all features to an feature vector X
# Only apda are shifted since we do not know what the atribute values will be in the future

def get_X(akda, apda, ahead):
    trend_df = featurize.t_attributes(apda.shift(ahead))
    apda_df = featurize.k_attributes(apda, ahead=ahead)
    merged = pd.concat([akda, trend_df, apda_df], axis=1)
    maxdate = pd.to_datetime((date.today() + timedelta(ahead-1)).strftime("%Y%m%d"))
    return merged[merged.index <= maxdate]


# Generate feature vector for each day ahead
# We do not produce forecasts in the weekends and on national holidays

def get_allX(akda, apda, datadir, maxahead=9):
    vddf = pd.read_csv(os.path.join(datadir,'vrije_dagen.csv'))
    svd = pd.to_datetime(vddf['Datum'])
    allX = dict()
    for ahead in range(1, maxahead + 1):
        key = 'X' + str(ahead)
        X = get_X(akda, apda, ahead)
        allX[key] = X[(X.index.weekday < 5) & (X.index.isin(svd) == False)].dropna()
    return allX


# Get Y vector for a specific team
# We do not produce forecasts in the weekends and on national holidays

def get_y(dates, ccd, team, datadir):
    vddf = pd.read_csv(os.path.join(datadir,'vrije_dagen.csv'))
    svd = pd.to_datetime(vddf['Datum'])
    df1 = ccd[team]
    df2 = df1.set_index(pd.to_datetime(df1.datum)).drop('datum', axis=1)
    df3 = df2[(df2.index.weekday < 5) & (df2.index.isin(svd) == False)]
    left = pd.DataFrame(index=dates)
    df3 = pd.merge(left=left, right=df3[['Aanbod']].replace(0, np.nan).dropna(), how='inner', left_index=True,
                   right_index=True)
    return df3
