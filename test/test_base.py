# Copyright 2023 Eurobios
# Licensed under the CeCILL License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://cecill.info/
import numpy as np
import pandas as pd
import pytest

from crep import merge, aggregate_constant, unbalanced_merge
from crep import tools
from crep.base import __fill_stretch, __merge


def test_merge_basic(get_examples):
    dfl, dfr = get_examples
    ret = merge(dfl, dfr,
                id_continuous=["t1", "t2"],
                id_discrete=["id"],
                how="outer")
    ret_l = merge(dfl, dfr,
                  id_continuous=["t1", "t2"],
                  id_discrete=["id"],
                  how="left")
    ret_i = merge(dfl, dfr,
                  id_continuous=["t1", "t2"],
                  id_discrete=["id"],
                  how="inner")
    ret_r = merge(dfl, dfr,
                  id_continuous=["t1", "t2"],
                  id_discrete=["id"],
                  how="right", verbose=True)
    ret_th = pd.DataFrame(
        dict(id=[1, 1, 1, 1, 2, 2, 2],
             t1=[0, 5, 10, 80, 0, 100, 120],
             t2=[5, 10, 80, 100, 90, 110, 130],
             data1=[0.2, 0.2, 0.2, 0.2, 0.1, 0.3, 0.2],
             data2=[np.nan, 0.2, 0.2, np.nan, 0.1, 0.3, 0.2],
             ))
    ret_th = ret_th.astype(ret.dtypes)
    ret_i_th = ret_th.dropna()
    ret_i_th.index = range(ret_i_th.__len__())
    assert ret.equals(ret_th)
    assert ret_l.equals(ret_th)
    assert ret_i.equals(ret_i_th)
    assert ret_r.equals(ret_i_th)


def test_fill_stretch(get_examples):
    dfl, _ = get_examples
    ret = __fill_stretch(dfl.__deepcopy__(),
                         id_continuous=["t1", "t2"],
                         id_discrete=["id"],
                         )
    assert ret["added"].sum() == 2


def test__merge(get_advanced_examples):
    df_left, df_right = get_advanced_examples

    df_merge = __merge(df_left, df_right,
                       id_discrete=["id"],
                       id_continuous=["t1", "t2"])


def test_admissible_data(get_advanced_examples):
    df_left, df_right = get_advanced_examples


def test_check_args(get_examples):
    pass


def test_aggregate_constant(get_examples):
    df1, _ = get_examples
    df1.index = np.random.uniform(size=len(df1))
    ret = aggregate_constant(df1, id_continuous=["t1", "t2"],
                             id_discrete=["id"])
    df_left = pd.DataFrame(
        dict(id=[1, 2, 2, 2],
             t1=[0, 0, 100, 120],
             t2=[100, 90, 110, 130],
             data1=[0.2, 0.1, 0.3, 0.2])
    )
    df_left.index = ret.index
    assert len(ret) < len(df1)
    assert df_left.equals(ret)


def test_merge_duplicates(get_examples):
    dfl, dfr = get_examples
    ret_r = merge(dfl, dfr,
                  id_continuous=["t1", "t2"],
                  id_discrete=["id"],
                  how="right", verbose=True, remove_duplicates=True)
    assert all(ret_r.isna().sum() == 0)


def test_aggregate_constant_no_aggregation(get_examples):
    _, dfr = get_examples
    ret = aggregate_constant(dfr, id_continuous=["t1", "t2"],
                             id_discrete=["id"])

    ret2 = aggregate_constant(ret, id_continuous=["t1", "t2"],
                              id_discrete=["id"])
    assert ret.equals(ret2)


def test_merge_symetry(get_examples):
    dfl, dfr = get_examples
    ret1 = merge(dfl, dfr,
                 id_continuous=["t1", "t2"],
                 id_discrete=["id"],
                 how="outer")
    ret2 = merge(dfr, dfl,
                 id_continuous=["t1", "t2"],
                 id_discrete=["id"],
                 how="outer")

    assert ret1.equals(ret2[ret1.columns])


def test_merge_discrete_id(get_advanced_examples):
    data_left, data_right = get_advanced_examples

    ret1 = merge(data_left, data_right,
                 id_continuous=["t1", "t2"],
                 id_discrete=["id", "id2"],
                 how="outer")
    ret2 = merge(data_left, data_right,
                 id_continuous=["t1", "t2"],
                 id_discrete=["id"],
                 how="outer")
    assert len(ret1) > len(ret2)


def test_build_admissible_dataset(get_advanced_examples):
    df_in = get_advanced_examples[1]
    data = {
        'id': [1, 2, 2, 3, 2, 2, 2, 2, 3, 3, 3, 3],
        't1': [5, 100, 120, 30, 0, 10, 10, 80, 10, 20, 20, 25],
        't2': [10, 110, 130, 35, 10, 80, 80, 90, 20, 25, 25, 30],
        'data2': [0.2, 0.3, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.3, 0.3, 0.2, 0.3]
    }

    # Creating the DataFrame
    df = pd.DataFrame(data)

    df_out = tools.build_admissible_data(df_in, id_continuous=["t1", "t2"],
                                         id_discrete=["id"])
    df_out = df_out.sort_values(by=["id", "t1", "t2", "data2"])
    df = df.sort_values(by=["id", "t1", "t2", 'data2'])
    df_out.index = df.index

    assert all(df["data2"].values == df_out["data2"].values)


def test_unbalanced_merge(get_advanced_examples):
    df_left, df_right = get_advanced_examples
    df_ret = unbalanced_merge(df_left, df_right,
                              id_discrete=["id"],
                              id_continuous=["t1", "t2"])


def test_merge_how(get_advanced_examples):
    data_left, data_right = get_advanced_examples
    with pytest.raises(Exception) as exc_info:
        merge(data_left, data_right,
              id_continuous=["t1", "t2"],
              id_discrete=["id", "id2"],
              how="fsdfjs")
    assert exc_info.value.args[0] == 'How must be in "left", "right", "inner", "outer"'

    with pytest.raises(Exception) as exc_info:
        merge(data_left, data_right,
              id_continuous=["t1", "t2", "id"],
              id_discrete=["id", "id2"],
              how="fsdfjs")
    assert exc_info.value.args[0] == "Only two continuous index is possible"

    with pytest.raises(Exception) as exc_info:
        merge(data_left, data_right,
              id_continuous=["t1", "t2", "t3"],
              id_discrete=["id", "id2"],
              how="fsdfjs")
    assert exc_info.value.args[0] == "t3 is not in columns"
