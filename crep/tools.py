# Copyright 2023 Eurobios
# Licensed under the CeCILL License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://cecill.info/
import numpy as np
import pandas as pd


def get_overlapping(data: pd.DataFrame,
                    id_discrete: iter,
                    id_continuous: iter) -> pd.Series:
    data_ = data.__deepcopy__()
    data_ = data_.sort_values(by=[*id_discrete, id_continuous[0]])
    data_["r1"] = range(len(data_))

    condition = data_[id_continuous[1]].iloc[:-1].values > data_[id_continuous[0]].iloc[1:].values
    data_["local_overlap"] = False
    data_.loc[data_.index[1:][condition], "local_overlap"] = True
    for i in id_discrete:
        values = data_[i]
        condition = values.iloc[:-1].values == values.iloc[1:].values
        condition = np.concatenate(([False], condition))
        data_["local_overlap"] &= condition

    data_ = data_.sort_values(by=[*id_discrete, id_continuous[1]])
    data_["r2"] = range(len(data_))
    data_ = data_.loc[data.index]
    overlap = pd.Series(data_["r1"] != data_["r2"]) | data_["local_overlap"]
    return overlap


def admissible_dataframe(data: pd.DataFrame,
                         id_discrete: iter,
                         id_continuous: iter):
    return sum(get_overlapping(data, id_discrete,
                               id_continuous)) == 0


def sample_non_admissible_data(data: pd.DataFrame,
                               id_discrete: iter,
                               id_continuous: iter) -> pd.DataFrame:
    return data[get_overlapping(data, id_discrete,
                                id_continuous)]


def build_admissible_data(df: pd.DataFrame, id_discrete: iter, id_continuous: iter) -> pd.DataFrame:
    df_non_admissible = sample_non_admissible_data(df, id_discrete, id_continuous).__deepcopy__()
    all_id_continuous = df_non_admissible[id_continuous[0]].to_list()
    all_id_continuous += df_non_admissible[id_continuous[1]].to_list()

    df_ret = pd.concat([df_non_admissible[id_discrete]]*2)
    df_ret[id_continuous[0]] = all_id_continuous
    df_ret["__disc__"] = compute_discontinuity(df_ret, id_discrete, id_continuous)
    df_ret = df_ret.sort_values(by=[*id_discrete, id_continuous[0]])
    df_ret[id_continuous[1]] = - df_ret[id_continuous[0]].diff(periods=-1) + df_ret[id_continuous[0]]
    df_ret = df_ret.dropna().drop(columns="__disc__")

    df_ret = pd.merge(df_ret,df_non_admissible, on=id_discrete, suffixes=("", "_tmp"))
    id_continuous_tmp = [str(i) + "_tmp" for i in id_continuous]
    c = df_ret[id_continuous[0]] < df_ret[id_continuous_tmp[1]]
    c &= df_ret[id_continuous[1]] > df_ret[id_continuous_tmp[0]]

    df_ret = df_ret.loc[c].drop(columns=id_continuous_tmp)
    df_ret = df_ret.astype(df_non_admissible.dtypes)

    df_ret_all = df[~get_overlapping(df, id_discrete, id_continuous)]
    df_ret_all = pd.concat((df_ret_all, df_ret))
    return df_ret_all


def compute_discontinuity(df, id_discrete, id_continuous):
    """
    Compute discontinuity in rail segment. The i-th element in return
    will be True if i-1 and i are discontinuous

    """
    discontinuity = np.zeros(len(df)).astype(bool)
    for col in id_discrete:
        if col in df.columns:
            discontinuity_temp = np.concatenate(
                ([False], df[col].values[1:] != df[col].values[:-1]))
            discontinuity |= discontinuity_temp

    if id_continuous[0] in df.columns and id_continuous[1] in df.columns:
        discontinuity_temp = np.concatenate(
            ([False], df[id_continuous[0]].values[1:] != df[id_continuous[
                1]].values[:-1]))
        discontinuity |= discontinuity_temp
    return discontinuity
