# Copyright 2023 Eurobios
# Licensed under the CeCILL License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://cecill.info/
import numpy as np
import pandas as pd





def build_admissible_data(df: pd.DataFrame, id_discrete: iter, id_continuous: iter) -> pd.DataFrame:
    df.index = range(len(df.index))
    df_non_admissible = sample_non_admissible_data(df, id_discrete, id_continuous).__deepcopy__()
    df_non_admissible = create_zones(df_non_admissible, id_discrete, id_continuous)

    df_non_admissible["__id__"] = range(len(df_non_admissible))
    df_add_non_admissible = df_non_admissible[[*id_discrete, *id_continuous, "__id__", "__zone__"]]
    all_id_continuous = df_non_admissible[id_continuous[0]].to_list()
    all_id_continuous += df_non_admissible[id_continuous[1]].to_list()

    # create good segmentation
    df_ret = pd.concat([df_non_admissible[[*id_discrete, "__zone__"]]] * 2)
    df_ret[id_continuous[0]] = all_id_continuous
    df_ret["__disc__"] = tools.compute_discontinuity(df_ret, id_discrete, id_continuous)
    df_ret = df_ret.sort_values(by=[*id_discrete, id_continuous[0]])
    df_ret[id_continuous[1]] = - df_ret[id_continuous[0]].diff(periods=-1) + df_ret[id_continuous[0]]
    df_ret = df_ret.dropna().drop(columns="__disc__")
    df_ret = df_ret.drop_duplicates().dropna()
    df_ret = df_ret[df_ret[id_continuous[1]] != df_ret[id_continuous[0]]]
    df_ret = df_ret.sort_values(by=[*id_discrete, id_continuous[0]])
    df_ret[id_continuous[1]] = df_ret[id_continuous[1]].astype(df[id_continuous[1]].dtype)

    df_ret = pd.merge(df_ret, df_add_non_admissible,
                      on=id_discrete + ["__zone__"], suffixes=("", "_tmp"))
    id_continuous_tmp = [str(i) + "_tmp" for i in id_continuous]
    c = df_ret[id_continuous[0]] < df_ret[id_continuous_tmp[1]]
    c &= df_ret[id_continuous[1]] > df_ret[id_continuous_tmp[0]]
    df_ret = df_ret.loc[c].drop(columns=id_continuous_tmp)

    df_ret = pd.merge(df_ret, df_non_admissible.drop(columns=[*id_discrete, *id_continuous, "__zone__"]), on="__id__"
                      ).drop(columns=["__id__", "__zone__"])
    df_ret = df_ret.astype(df_non_admissible.dtypes.drop(["__id__", "__zone__"]))

    df_ret_all = df[~get_overlapping(df, id_discrete, id_continuous)]
    df_ret_all = pd.concat((df_ret_all, df_ret))
    df_ret_all = df_ret_all.sort_values(by=[*id_discrete, id_continuous[0]])
    df_ret_all.index = range(len(df_ret_all.index))
    return df_ret_all


def create_zones(df: pd.DataFrame, id_discrete: iter, id_continuous: iter):
    df_idx = df[[*id_discrete, *id_continuous]]
    df_idx = df_idx.sort_values([*id_discrete, id_continuous[1]])
    df_idx["__zf__"] = range(len(df_idx))
    df_idx = df_idx.sort_values([*id_discrete, id_continuous[0]])
    df_idx["__zi__"] = range(len(df_idx))
    c_zone = (df_idx["__zf__"] - df_idx["__zi__"]) == 0

    df_idx["__id2_prev__"] = df[id_continuous[1]]
    df_idx.loc[df_idx.index[1:], "__id2_prev__"] = df_idx.loc[df_idx.index[:-1], id_continuous[1]].values
    c_inner = df_idx[id_continuous[0]] >= df_idx["__id2_prev__"]

    df_idx["__zone__"] = c_zone & c_inner
    df_idx["__zone__"] = df_idx["__zone__"].cumsum()
    df = pd.merge(df, df_idx.drop(columns=["__zi__", "__zf__", '__id2_prev__']), on=[*id_discrete, *id_continuous])
    return df

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
    df.index = range(len(df.index))
    df_non_admissible = sample_non_admissible_data(df, id_discrete, id_continuous).__deepcopy__()
    df_non_admissible = create_zones(df_non_admissible, id_discrete, id_continuous)

    df_non_admissible["__id__"] = range(len(df_non_admissible))
    df_add_non_admissible = df_non_admissible[[*id_discrete, *id_continuous, "__id__", "__zone__"]]
    all_id_continuous = df_non_admissible[id_continuous[0]].to_list()
    all_id_continuous += df_non_admissible[id_continuous[1]].to_list()

    df_ret = pd.concat([df_non_admissible[[*id_discrete, "__zone__"]]]*2)
    df_ret[id_continuous[0]] = all_id_continuous
    df_ret["__disc__"] = compute_discontinuity(df_ret, id_discrete, id_continuous)
    df_ret = df_ret.sort_values(by=[*id_discrete, id_continuous[0]])
    df_ret[id_continuous[1]] = - df_ret[id_continuous[0]].diff(periods=-1) + df_ret[id_continuous[0]]
    df_ret = df_ret.dropna().drop(columns="__disc__")
    df_ret = df_ret.drop_duplicates().dropna()

    df_ret = pd.merge(df_ret, df_add_non_admissible,
                      on=id_discrete + ["__zone__"], suffixes=("", "_tmp"))
    id_continuous_tmp = [str(i) + "_tmp" for i in id_continuous]
    c = df_ret[id_continuous[0]] < df_ret[id_continuous_tmp[1]]
    c &= df_ret[id_continuous[1]] > df_ret[id_continuous_tmp[0]]
    df_ret = df_ret.loc[c].drop(columns=id_continuous_tmp)

    df_ret = pd.merge(df_ret, df_non_admissible.drop(columns=[*id_discrete, *id_continuous, "__zone__"]), on="__id__"
                      ).drop(columns=["__id__", "__zone__"])
    df_ret = df_ret.astype(df_non_admissible.dtypes.drop(["__id__", "__zone__"]))

    df_ret_all = df[~get_overlapping(df, id_discrete, id_continuous)]
    df_ret_all = pd.concat((df_ret_all, df_ret))
    return df_ret_all


def create_zones(df: pd.DataFrame, id_discrete: iter, id_continuous: iter):
    df_idx = df[[*id_discrete, *id_continuous]]
    df_idx = df_idx.sort_values([*id_discrete, id_continuous[1]])
    df_idx["__zf__"] = range(len(df_idx))
    df_idx = df_idx.sort_values([*id_discrete, id_continuous[0]])
    df_idx["__zi__"] = range(len(df_idx))
    df_idx["__zone__"] = df_idx["__zf__"] == df_idx["__zi__"]
    df_idx["__zone__"] = df_idx["__zone__"].cumsum()

    df = pd.merge(df, df_idx.drop(columns=["__zi__", "__zf__"]), on=[*id_discrete, *id_continuous])
    return df


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
