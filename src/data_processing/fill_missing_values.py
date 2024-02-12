import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


def replace_value_by_nn(time_windows):
    """Impute missing values using the k-Nearest Neighbors algorithm.
       i.e. new values will be selected from the k closest time windows.
    """
    imputer = KNNImputer(n_neighbors=3)
    return imputer.fit_transform(np.array(time_windows).squeeze(-1))


def fill_empty_days(load, day_size=48, load_feature_name="load"):
    """fill empty days with day values from previous/next week"""
    st = 0
    en = day_size
    while en < len(load):
        day_serie = load[load_feature_name].values[st:en]
        if np.isnan(day_serie).sum() == day_size:
            empty_day = True
            prev_week_day_st = st - day_size*7
            while prev_week_day_st>0:
                prev_week_day_serie = load[load_feature_name].values[prev_week_day_st:prev_week_day_st+day_size]
                if np.isnan(prev_week_day_serie).sum() != day_size:
                    load[load_feature_name].values[st:en] = prev_week_day_serie
                    empty_day = False
                    break
                prev_week_day_st -= day_size*7
            
            next_week_day_st = st + day_size*7
            while en<len(load) and empty_day:
                next_week_day_serie = load[load_feature_name].values[next_week_day_st:next_week_day_st+day_size]
                if np.isnan(next_week_day_serie).sum() != day_size:
                    load[load_feature_name].values[st:en] = next_week_day_serie
                    empty_day = False
                    break
                next_week_day_st += day_size*7

            if empty_day:
                print("Couldn't find a non-empty day to fill empty day at index" , st, " with previous/next week.")

        st += day_size
        en += day_size
    
    return load


def fill_missing_values(load, day_size):
    """Replace missing days by previous/next week and missing values by KNN imputation"""
    idx = pd.date_range(load.index[0], load.index[-1], freq="30T")
    load = load.reindex(idx, fill_value=np.nan)

    # fill empty days
    load_feature_name = load.columns[0]
    load = fill_empty_days(load, day_size, load_feature_name)

    # fill missing values in days
    windows = [load.iloc[i:i+day_size] for i in range(0, len(load)-day_size+1, day_size)]
    filled_windows = replace_value_by_nn(windows)
    load = pd.DataFrame(np.concatenate(filled_windows), columns=[load_feature_name], index=load.index[:len(filled_windows)*day_size])

    return load
