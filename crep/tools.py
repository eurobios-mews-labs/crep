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
                               id_continuous: iter)->pd.DataFrame:
    return data[get_overlapping(data, id_discrete,
                                id_continuous)]
