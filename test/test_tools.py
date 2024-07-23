# Copyright 2023 Eurobios
# Licensed under the CeCILL License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://cecill.info/

import pandas as pd
import numpy as np

from crep.tools import get_overlapping, admissible_dataframe, sample_non_admissible_data, create_zones, \
    build_admissible_data

id_discrete, id_continuous = ["id", "id2"], ["t1", "t2"]
data = {
    'id': [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, np.nan],
    'id2': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    't1': [932, 996, 2395, 2395, 3033, 3628, 4126, 4140, 4154, 63, 63, 271, 271, np.nan],
    't2': [2395, 2324, 3033, 3628, 3035, 4140, 4140, 5508, 5354, 199, 199, 351, 357, np.nan],
    'LONGUEUR': [1463, 1328, 638, 1233, 2, 512, 14, 1368, 1200, 136, 208, 935, 1086, np.nan],
    '__zone__': [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5]
}
data = pd.DataFrame(data).sort_values(by=[*id_discrete, *id_continuous])


def test_no_overlapping(get_examples):
    assert sum(get_overlapping(
        get_examples[0], ["id"], ["t1", "t2"]
    )) == 0
    assert admissible_dataframe(get_examples[0], ["id"], id_continuous)


def test_overlapping(get_examples):
    df = get_examples[0]
    df.loc[1, "t1"] = 5
    ret = get_overlapping(df, ["id"], id_continuous)
    assert sum(ret) == 2


def test_sample_overlapping(get_examples):
    df = get_examples[0]
    df.loc[1, "t1"] = 5
    ret = sample_non_admissible_data(df, ["id"], id_continuous)
    assert ret.equals(df.loc[[0, 1]])


def test_create_zones():
    df = data.sort_values(by=[*id_discrete, *id_continuous])
    df_out = create_zones(df.drop(columns="__zone__"), id_discrete=id_discrete, id_continuous=id_continuous)
    df_out = df_out.sort_values(by=[*id_discrete, *id_continuous])
    assert all(df_out["__zone__"].values == df["__zone__"].values)


def test_build_admissible_data_frame():
    ret = build_admissible_data(df=data.drop(columns="__zone__"), id_discrete=id_discrete, id_continuous=id_continuous)
    ret = ret.drop_duplicates(subset=[*id_discrete, *id_continuous])

    assert admissible_dataframe(ret, id_discrete=id_discrete, id_continuous=id_continuous)


def test_fix_point_non_admissible(get_examples):

    df_non_admissible = sample_non_admissible_data(data, id_discrete, id_continuous)
    df_non_admissible_2 = sample_non_admissible_data(df_non_admissible, id_discrete, id_continuous)

    assert df_non_admissible.equals(df_non_admissible_2)

    df_non_admissible = sample_non_admissible_data(get_examples[0], ['id'], id_continuous)
    df_non_admissible_2 = sample_non_admissible_data(df_non_admissible, ['id'], id_continuous)

    assert df_non_admissible.equals(df_non_admissible_2)

    df_non_admissible = sample_non_admissible_data(get_examples[1], ['id'], id_continuous)
    df_non_admissible_2 = sample_non_admissible_data(df_non_admissible, ['id'], id_continuous)

    assert df_non_admissible.equals(df_non_admissible_2)
