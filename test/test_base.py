# Copyright 2023 Eurobios
# Licensed under the CeCILL License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://cecill.info/
import numpy as np
import pandas as pd
import pytest

from crep import base
from crep import (merge, aggregate_constant, unbalanced_merge, unbalanced_concat, homogenize_within,
                  aggregate_duplicates)
from crep import tools


def test_merge_basic(get_examples):
    df_left, df_right = get_examples
    ret = merge(df_left, df_right,
                id_continuous=["t1", "t2"],
                id_discrete=["id"],
                how="outer")
    ret_l = merge(df_left, df_right,
                  id_continuous=["t1", "t2"],
                  id_discrete=["id"],
                  how="left")
    ret_i = merge(df_left, df_right,
                  id_continuous=["t1", "t2"],
                  id_discrete=["id"],
                  how="inner")
    ret_r = merge(df_left, df_right,
                  id_continuous=["t1", "t2"],
                  id_discrete=["id"],
                  how="right", verbose=True)
    ret_th = pd.DataFrame({
        "id": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        "t1": [0, 5, 10, 80, 0, 80, 100, 110, 120, 130, 135],
        "t2": [5, 10, 80, 100, 80, 90, 110, 120, 130, 135, 145],
        "data1": [0.4, 0.4, 0.3, 0.3, 0.1, 0.1, 0.3, np.nan, 0.2, 0.2, 0.2],
        "data2": [np.nan, 0.25, 0.25, np.nan, 0.15, np.nan, 0.35, 0.35, 0.35, 0.50, np.nan]
    })
    ret_th = ret_th.astype(ret.dtypes)
    ret_i_th = ret_th.dropna()
    ret_th_l = ret_th.dropna(subset=["data1"])
    ret_th_r = ret_th.dropna(subset=["data2"])

    ret_i_th.index = range(ret_i_th.__len__())
    ret_th_l.index = range(ret_th_l.__len__())
    ret_th_r.index = range(ret_th_r.__len__())

    assert ret.equals(ret_th)
    assert ret_l.equals(ret_th_l)
    assert ret_i.equals(ret_i_th)
    assert ret_r.equals(ret_th_r)


def test__merge(get_advanced_examples):
    df_left, df_right = get_advanced_examples

    df_merge = base.__merge(df_left, df_right,
                            id_discrete=["id"],
                            id_continuous=["t1", "t2"])


def test_admissible_data(get_advanced_examples):
    df_left, df_right = get_advanced_examples


def test_check_args(get_examples):
    pass


def test_aggregate_constant(get_examples):
    df_left = get_examples[0].__deepcopy__()
    df_left["data1"] = 1
    df_left.index = np.random.uniform(size=len(df_left))
    ret = aggregate_constant(df_left, id_continuous=["t1", "t2"],
                             id_discrete=["id"])
    ret_th = pd.DataFrame({'id': [1, 2, 2, 2],
                           't1': [0, 0, 100, 120],
                           't2': [100, 90, 110, 145],
                           'data1': [1, 1, 1, 1]})
    ret_th.index = ret.index
    assert len(ret) < len(df_left)
    assert ret_th.equals(ret)


def test_merge_duplicates(get_examples):
    df_left, df_right = get_examples
    ret_r = merge(df_left, df_right,
                  id_continuous=["t1", "t2"],
                  id_discrete=["id"],
                  how="inner", verbose=True, remove_duplicates=True)
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

    df_out = tools.build_admissible_data(df_in, id_continuous=["t1", "t2"],
                                         id_discrete=["id"])
    df_out = df_out.drop_duplicates(subset=["id", "t1", "t2"])
    print(df_out)
    ret = tools.admissible_dataframe(df_out, id_continuous=["t1", "t2"],
                                     id_discrete=["id"])

    assert ret
    df_out_out = tools.build_admissible_data(df_out, id_continuous=["t1", "t2"],
                                             id_discrete=["id"])
    assert df_out_out.equals(df_out)


def test_unbalanced_merge(get_advanced_examples):
    # FIXME correct this
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


def test_merge_event(get_examples):
    df_left, df_right = get_examples
    base.merge_event(data_left=df_left, data_right=df_right, id_discrete=["id"],
                     id_continuous=["t1", "t2"],
                     id_event="t1"
                     )


def test_regular_table(get_examples):
    df_left, df_right = get_examples
    df_ret = base.create_regular_segmentation(
        df_left, length=9, id_discrete=["id"], id_continuous=["t1", "t2"]
    )
    length = df_ret["t2"] - df_ret['t1']
    assert np.var(length - 9) < 1

    df_ret = base.create_regular_segmentation(
        df_left, length=0, id_discrete=["id"], id_continuous=["t1", "t2"]
    )
    assert df_ret.equals(df_left)


def test_merge_event_case2():
    df1 = pd.DataFrame({"id": [1000] * 4 + [2000] * 2 + [4000] * 2,
                        "cont1": [50, 100, 150, 200, 50, 100, 250, 300],
                        "cont2": [100, 150, 200, 250, 100, 150, 300, 350],
                        "date": list(range(2000, 2016, 2)),
                        })
    df2 = pd.DataFrame({"id": [1000] * 4 + [2000] * 3 + [3000],
                        "cont1": [60, 160, 160, 160, 60, 110, 110, 110],
                        "cont2": [110, 230, 230, 230, 110, 360, 360, 160],
                        "ev": [80, 180, 190, 220, 70, 140, 320, 125],
                        "value": list(range(0, 8))
                        })
    df_test = base.merge_event(
        data_left=df1,
        data_right=df2,
        id_event="ev",
        id_discrete=["id"],
        id_continuous=["cont1", "cont2"]
    )
    assert str(df_test) == str(pd.DataFrame(
        {'id': [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 2000, 2000, 2000, 2000, 4000, 4000],
         'cont1': [50.0, 50.0, 100.0, 150.0, 150.0, 150.0, 200.0, 200.0, 50.0, 50.0, 100.0, 100.0, 250.0, 300.0],
         'cont2': [100.0, 100.0, 150.0, 200.0, 200.0, 200.0, 250.0, 250.0, 100.0, 100.0, 150.0, 150.0, 300.0,
                   350.0],
         'date': [2000.0, 2000.0, 2002.0, 2004.0, 2004.0, 2004.0, 2006.0, 2006.0, 2008.0, 2008.0, 2010.0, 2010.0,
                  2012.0, 2014.0],
         'ev': [np.nan, 80.0, np.nan, np.nan, 180.0, 190.0, np.nan, 220.0, np.nan, 70.0, np.nan, 140.0, np.nan,
                np.nan],
         'value': [np.nan, 0.0, np.nan, np.nan, 1.0, 2.0, np.nan, 3.0, np.nan, 4.0, np.nan, 5.0, np.nan, np.nan]}
    )), "\n" + str(df_test)


def test_aggregate_continuous_data_case1():
    df = pd.DataFrame({"discr1": [1000] * 8 + [2000] * 4,
                       "discr2": [1] * 4 + [2] * 4 + [1] * 4,
                       "cont1": [50, 100, 150, 200, 40, 75, 135, 178, 50, 90, 150, 210],
                       "cont2": [100, 150, 200, 250, 75, 135, 178, 211, 90, 150, 210, 280],
                       "date": list(range(2000, 2024, 2))})
    df_test = base.aggregate_continuous_data(
        df=df,
        id_discrete=["discr1", "discr2"],
        id_continuous=["cont1", "cont2"],
        dict_agg={"sum": ["date"], "mean": ['date']},
        target_size=140,
        verbose=True
    )
    assert \
        ((df_test["sum_date"].to_list() == [4002, 4010, 8044, 4034, 4042])
         & (list(np.round(df_test["mean_date"].to_list(), 1)) == [2001.0, 2005.0, 2010.9, 2017.2, 2021.1])), \
        "\n" + str(df_test)


def test_aggregate_continuous_data_case2():
    df = pd.DataFrame({"discr1": [830341] * 4,
                       "discr2": ["v2"] * 4,
                       "cont1": [637, 704, 634008, 634062],
                       "cont2": [704, 789, 634062, 634130],
                       "date": [2000, 2002, 2004, 2006]})
    df_test = base.aggregate_continuous_data(
        df=df,
        id_discrete=["discr1", "discr2"],
        id_continuous=["cont1", "cont2"],
        dict_agg={"sum": ["date"], "mean": ['date']},
        target_size=216
    )
    assert \
        ((df_test["sum_date"].to_list() == [4002, 4010])
         & (list(np.round(df_test["mean_date"].to_list(), 1)) == [2001.1, 2005.1])), \
        "\n" + str(df_test)


def test_split_segment():
    df = pd.DataFrame({"discr1": [1000] * 5 + [2000] * 3,
                       "discr2": [1] * 2 + [2] * 3 + [1] * 3,
                       "cont1": [50, 100, 40, 75, 300, 50, 90, 250],
                       "cont2": [100, 150, 75, 300, 380, 90, 250, 310]})
    df_test = base.split_segment(
        df=df,
        id_discrete=["discr1", "discr2"],
        id_continuous=["cont1", "cont2"],
        target_size=50, verbose=True
    )
    assert \
        ((df_test["cont1"].to_list() == [50, 100, 40, 75, 120, 165, 210, 255, 300, 340, 50, 90, 143, 197, 250])
         & (df_test["cont2"].to_list() == [100, 150, 75, 120, 165, 210, 255, 300, 340, 380, 90, 143, 197, 250, 310])), \
        "\n" + str(df_test)


def test_segmentation_regular(get_examples):
    df_left, df_right = get_examples
    id_continuous = ["t1", "t2"]
    df = df_right
    length_target = 7
    length_minimal = 30
    id_discrete = ["id"]
    ret = base.segmentation_regular(
        df,
        id_discrete,
        id_continuous,
        length_target,
        length_minimal,
    )
    length = ret["t2"] - ret["t1"]
    assert np.abs(length.mean() - length_target).std() < 0.01


def test_aggregate(get_examples):
    df_left, df_right = get_examples
    id_continuous = ["t1", "t2"]
    df = df_right
    length_target = 7
    length_minimal = 30
    id_discrete = ["id"]
    df_target_segmentation = base.segmentation_regular(
        df,
        id_discrete,
        id_continuous,
        length_target,
        length_minimal,
    )
    ret = base.aggregate(df, df_target_segmentation,
                         id_discrete=id_discrete,
                         id_continuous=id_continuous)


def test_homogenize_within_case1():
    # I - df["__diff__"].min() = 33 < 54 and agg_applicable, so it should apply method "agg"
    df = pd.DataFrame({"discr1": [1000] * 8 + [2000] * 4,
                       "discr2": [1] * 4 + [2] * 4 + [1] * 4,
                       "cont1": [50, 100, 150, 200, 40, 75, 135, 178, 50, 90, 150, 210],
                       "cont2": [100, 150, 200, 250, 75, 135, 178, 211, 90, 150, 210, 280],
                       "date": list(range(2000, 2024, 2))})
    df_test = homogenize_within(
        df=df,
        id_discrete=["discr1", "discr2"],
        id_continuous=["cont1", "cont2"],
        target_size=140,
        dict_agg={"min": ["cont1"], "max": ["cont2"], "sum": ["date"], "mean": ['date']},
    )
    assert \
        len(df_test) < len(df), \
        "\n" + str(df_test)


def test_homogenize_within_case2():
    # II - method split but target_size > (df["__diff__"].min() * 1.33) so target size should be reduced to int(33 * 1.33)
    df = pd.DataFrame({"discr1": [1000] * 8 + [2000] * 4,
                       "discr2": [1] * 4 + [2] * 4 + [1] * 4,
                       "cont1": [50, 100, 150, 200, 40, 75, 135, 178, 50, 90, 150, 210],
                       "cont2": [100, 150, 200, 250, 75, 135, 178, 211, 90, 150, 210, 280],
                       "date": list(range(2000, 2024, 2))})
    df_test = homogenize_within(
        df=df,
        id_discrete=["discr1", "discr2"],
        id_continuous=["cont1", "cont2"],
        method="split",
        target_size=100,
    )
    diff = df_test["cont2"] - df_test["cont1"]
    assert \
        ((len(df_test) >= len(df))
         & (diff.max() / diff.min() < 2)), \
        "\n" + str(df_test)


def test_homogenize_within_case3():
    # III - method split without target size
    df = pd.DataFrame({"discr1": [1000] * 8 + [2000] * 4,
                       "discr2": [1] * 4 + [2] * 4 + [1] * 4,
                       "cont1": [50, 100, 150, 200, 40, 75, 135, 178, 50, 90, 150, 210],
                       "cont2": [100, 150, 200, 250, 75, 135, 178, 211, 90, 150, 210, 280],
                       "date": list(range(2000, 2024, 2))})
    df_test = homogenize_within(
        df=df,
        id_discrete=["discr1", "discr2"],
        id_continuous=["cont1", "cont2"],
        method="split"
    )
    diff = df_test["cont2"] - df_test["cont1"]
    assert \
        ((len(df_test) >= len(df))
         & (diff.max() / diff.min() < 2)), \
        "\n" + str(df_test)


def test_homogenize_within_case4():
    # IV - method split and agg, without target size. No dict_agg, so default agg should be mean.
    df = pd.DataFrame({"discr1": [1000] * 5 + [2000] * 3,
                       "discr2": [1] * 2 + [2] * 3 + [1] * 3,
                       "cont1": [50, 100, 40, 75, 300, 50, 90, 250],
                       "cont2": [100, 150, 75, 300, 380, 90, 250, 310],
                       "date": [2000, 2002, 2003, 2006, 2008, 2010, 2012, 2014]})
    df_test = homogenize_within(
        df=df,
        id_discrete=["discr1", "discr2"],
        id_continuous=["cont1", "cont2"],
        method=["agg", "split"],
        target_size=100,
    )
    diff = df_test["cont2"] - df_test["cont1"]
    assert \
        ((len(df_test) >= len(df))
         & (diff.max() / diff.min() < 2)), \
        "\n" + str(df_test)


def test_homogenize_within_case5():
    # V - method split and agg, without target size. Both should be applied
    df = pd.DataFrame({"discr1": [1000] * 5 + [2000] * 3,
                       "discr2": [1] * 2 + [2] * 3 + [1] * 3,
                       "cont1": [50, 100, 40, 75, 300, 50, 90, 250],
                       "cont2": [100, 150, 75, 300, 380, 90, 250, 310]})
    df_test = homogenize_within(
        df=df,
        id_discrete=["discr1", "discr2"],
        id_continuous=["cont1", "cont2"],
        method=["agg", "split"],
        dict_agg={}
    )
    diff = df_test["cont2"] - df_test["cont1"]
    diff0 = df["cont2"] - df["cont1"]
    assert \
        ((diff.min() > diff0.min())  # proof of agg
         & (diff.max() < diff0.max())  # proof of split
         & (diff.max() / diff.min() < 2)), \
        "\n" + str(df_test)


def test_homogenize_within_case6():
    # VI - method split and agg, with target size. Both should be applied
    df5 = pd.DataFrame({"discr1": [1000] * 5 + [2000] * 3,
                        "discr2": [1] * 2 + [2] * 3 + [1] * 3,
                        "cont1": [50, 100, 40, 75, 300, 50, 90, 250],
                        "cont2": [100, 150, 75, 300, 380, 90, 250, 310]})
    df_test = homogenize_within(
        df=df5,
        id_discrete=["discr1", "discr2"],
        id_continuous=["cont1", "cont2"],
        method=["agg", "split"],
        target_size=50,
        dict_agg={}
    )
    diff = df_test["cont2"] - df_test["cont1"]
    assert \
        ((diff.max() / diff.min() < 2)  # overall wanted result
         & (diff.max() < 100)), \
        "\n" + str(df_test)


def test_homogenize_between():
    df1 = pd.DataFrame({"discr1": [1000] * 4,
                        "discr2": [1] * 4,
                        "cont1": [50, 110, 155, 200],
                        "cont2": [110, 155, 200, 260]})
    df2 = pd.DataFrame({"discr1": [1000] * 2,
                        "discr2": [1] * 2,
                        "cont1": [50, 200],
                        "cont2": [200, 260]})
    df_test = base.homogenize_between(
        df1=df1,
        df2=df2,
        id_discrete=["discr1", "discr2"],
        id_continuous=["cont1", "cont2"]
    )
    diff0 = df_test[0]["cont2"] - df_test[0]["cont1"]
    diff1 = df_test[1]["cont2"] - df_test[1]["cont1"]
    assert \
        ((diff0.max() / diff1.min() < 2)
         & (diff1.max() / diff0.min() < 2)), \
        "\n" + str(df_test[0]) + "\n" + str(df_test[1])


def test_aggregate_duplicates_case1():
    df = pd.DataFrame({"discr1": [1000] * 4 + [2000] * 5,
                       "discr2": [1] * 2 + [2] * 2 + [1] * 2 + [2] * 3,
                       "cont1": [50, 80, 80, 80, 80, 150, 80, 80, 80],
                       "cont2": [80, 150, 150, 150, 150, 250, 105, 105, 105],
                       "date": list(range(2000, 2018, 2))})
    df_test = aggregate_duplicates(
        df=df,
        id_discrete=["discr1", "discr2"],
        id_continuous=["cont1", "cont2"], verbose=True
    )
    assert str(df_test) == str(pd.DataFrame({
        'discr1': [1000, 1000, 1000, 2000, 2000, 2000],
        'discr2': [1, 1, 2, 1, 1, 2],
        'cont1': [50, 80, 80, 80, 150, 80],
        'cont2': [80, 150, 150, 150, 250, 105],
        'mean_date': [2000.0, 2002.0, 2005.0, 2008.0, 2010.0, 2014.0]
    })), "\n" + str(df_test)


def test_aggregate_duplicates_case2():
    df = pd.DataFrame({"discr1": [1000] * 4 + [2000] * 5,
                       "discr2": [1] * 2 + [2] * 2 + [1] * 2 + [2] * 3,
                       "cont1": [50, 80, 80, 80, 80, 150, 80, 80, 80],
                       "cont2": [80, 150, 150, 150, 150, 250, 105, 105, 105],
                       "mean_date": list(range(2000, 2018, 2))})
    df_test = aggregate_duplicates(
        df=df,
        id_discrete=["discr1", "discr2"],
        id_continuous=["cont1", "cont2"],
    )
    assert str(df_test) == str(pd.DataFrame({
        'discr1': [1000, 1000, 1000, 2000, 2000, 2000],
        'discr2': [1, 1, 2, 1, 1, 2],
        'cont1': [50, 80, 80, 80, 150, 80],
        'cont2': [80, 150, 150, 150, 250, 105],
        'mean_date': [2000.0, 2002.0, 2005.0, 2008.0, 2010.0, 2014.0]
    })), "\n" + str(df_test)


def test_aggregate_duplicates_case3():
    df = pd.DataFrame({"discr1": [1000] * 4 + [2000] * 5,
                       "discr2": [1] * 2 + [2] * 2 + [1] * 2 + [2] * 3,
                       "cont1": [50, 80, 80, 80, 80, 150, 80, 80, 80],
                       "cont2": [80, 150, 150, 150, 150, 250, 105, 105, 105],
                       "mean_date": list(range(2000, 2018, 2))})
    dict_agg = {"mean": ["mean_date"], "max": ["mean_date"]}
    df_test = aggregate_duplicates(
        df=df,
        id_discrete=["discr1", "discr2"],
        id_continuous=["cont1", "cont2"],
        dict_agg=dict_agg
    )
    assert str(df_test) == str(pd.DataFrame({
        'discr1': [1000, 1000, 1000, 2000, 2000, 2000],
        'discr2': [1, 1, 2, 1, 1, 2],
        'cont1': [50, 80, 80, 80, 150, 80],
        'cont2': [80, 150, 150, 150, 250, 105],
        'mean_date': [2000.0, 2002.0, 2005.0, 2008.0, 2010.0, 2014.0],
        'max_date': [2000, 2002, 2006, 2008, 2010, 2016]
    })), "\n" + str(df_test)


# ====================================================================================================================
#                                                   UNBALANCED CONCAT
# ====================================================================================================================


def test_unbalanced_concat_case1():
    # row 2 count_0: df1 == df2, row 2 count_1: df1 < df2,
    df1 = pd.DataFrame({"discr": [1, 1, 1],
                        "cont_0": [0, 50, 100],
                        "cont_1": [50, 100, 200]})
    df2 = pd.DataFrame({"discr": [1, 1, 1],
                        "cont_0": [0, 50, 50],
                        "cont_1": [50, 200, 200],
                        "other": [1, 2, 3]})
    df_test = unbalanced_concat(
        df1=df1,
        df2=df2,
        id_discrete=["discr"],
        id_continuous=["cont_0", "cont_1"]
    )
    assert str(df_test) == str(pd.DataFrame({
        'discr': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'cont_0': [0.0, 0.0, 50.0, 50.0, 50.0, 100.0, 100.0, 100.0, 125.0, 125.0, 125.0, 150.0, 150.0, 150.0],
        'cont_1': [50.0, 50.0, 100.0, 100.0, 100.0, 125.0, 125.0, 125.0, 150.0, 150.0, 150.0, 200.0, 200.0, 200.0],
        'other': [np.nan, 1.0, np.nan, 2.0, 3.0, np.nan, 2.0, 3.0, np.nan, 2.0, 3.0, np.nan, 2.0, 3.0]
    })), "\n" + str(df_test)


def test_unbalanced_concat_case2():
    # row 2 count_0: df1 > df2, row 2 count_1: df1 == df2,
    df1 = pd.DataFrame({"discr": [1, 1],
                        "cont_0": [0, 100],
                        "cont_1": [100, 150]})
    df2 = pd.DataFrame({"discr": [1, 1, 1],
                        "cont_0": [0, 50, 50],
                        "cont_1": [50, 150, 150],
                        "other": [1, 2, 3]})
    df_test = unbalanced_concat(
        df1=df1,
        df2=df2,
        id_discrete=["discr"],
        id_continuous=["cont_0", "cont_1"]
    )
    assert str(df_test) == str(pd.DataFrame({
        'discr': [1, 1, 1, 1, 1, 1, 1, 1],
        'cont_0': [0.0, 0.0, 50.0, 50.0, 50.0, 100.0, 100.0, 100.0],
        'cont_1': [50.0, 50.0, 100.0, 100.0, 100.0, 150.0, 150.0, 150.0],
        'other': [np.nan, 1.0, np.nan, 2.0, 3.0, np.nan, 2.0, 3.0]
    })), "\n" + str(df_test)


def test_unbalanced_concat_case3():
    # row 2 count_0: df1 < df2, row 2 count_1: df1 == df2,
    df1 = pd.DataFrame({"discr": [1, 1],
                        "cont_0": [0, 50],
                        "cont_1": [50, 150]})
    df2 = pd.DataFrame({"discr": [1, 1, 1],
                        "cont_0": [0, 100, 100],
                        "cont_1": [100, 150, 150],
                        "other": [1, 2, 3]})
    df_test = unbalanced_concat(
        df1=df1,
        df2=df2,
        id_discrete=["discr"],
        id_continuous=["cont_0", "cont_1"]
    )
    assert str(df_test) == str(pd.DataFrame({
        'discr': [1, 1, 1, 1, 1, 1, 1],
        'cont_0': [0.0, 0.0, 50.0, 50.0, 100.0, 100.0, 100.0],
        'cont_1': [50.0, 50.0, 100.0, 100.0, 150.0, 150.0, 150.0],
        'other': [np.nan, 1.0, np.nan, 1.0, np.nan, 2.0, 3.0]
    })), "\n" + str(df_test)


def test_unbalanced_concat_case4():
    # row 2 count_0: df1 < df2, row 2 count_1: df1 < df2,
    df1 = pd.DataFrame({"discr": [1, 1, 1],
                        "cont_0": [0, 50, 150],
                        "cont_1": [50, 150, 200]})
    df2 = pd.DataFrame({"discr": [1, 1, 1],
                        "cont_0": [0, 100, 100],
                        "cont_1": [100, 200, 200],
                        "other": [1, 2, 3]})
    df_test = unbalanced_concat(
        df1=df1,
        df2=df2,
        id_discrete=["discr"],
        id_continuous=["cont_0", "cont_1"]
    )
    assert str(df_test) == str(pd.DataFrame(
        {'discr': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         'cont_0': [0.0, 0.0, 50.0, 50.0, 100.0, 100.0, 100.0, 150.0, 150.0, 150.0],
         'cont_1': [50.0, 50.0, 100.0, 100.0, 150.0, 150.0, 150.0, 200.0, 200.0, 200.0],
         'other': [np.nan, 1.0, np.nan, 1.0, np.nan, 2.0, 3.0, np.nan, 2.0, 3.0]
         })), "\n" + str(df_test)


def test_unbalanced_concat_case5():
    # row 2 count_0: df1 > df2, row 2 count_1: df1 > df2,
    df1 = pd.DataFrame({"discr": [1, 1],
                        "cont_0": [0, 100],
                        "cont_1": [100, 200]})
    df2 = pd.DataFrame({"discr": [1, 1, 1, 1],
                        "cont_0": [0, 50, 50, 150],
                        "cont_1": [50, 150, 150, 200],
                        "other": [1, 2, 3, 4]})
    df_test = unbalanced_concat(
        df1=df1,
        df2=df2,
        id_discrete=["discr"],
        id_continuous=["cont_0", "cont_1"]
    )
    assert str(df_test) == str(pd.DataFrame({
        'discr': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'cont_0': [0.0, 0.0, 50.0, 50.0, 50.0, 100.0, 100.0, 100.0, 150.0, 150.0],
        'cont_1': [50.0, 50.0, 100.0, 100.0, 100.0, 150.0, 150.0, 150.0, 200.0, 200.0],
        'other': [np.nan, 1.0, np.nan, 2.0, 3.0, np.nan, 2.0, 3.0, np.nan, 4.0]
    })), "\n" + str(df_test)


def test_unbalanced_concat_case6():
    # row 2 count_0: df1 > df2, row 2 count_1: df1 < df2,
    df1 = pd.DataFrame({"discr": [1, 1, 1],
                        "cont_0": [0, 100, 150],
                        "cont_1": [100, 150, 200]})
    df2 = pd.DataFrame({"discr": [1, 1, 1],
                        "cont_0": [0, 50, 50],
                        "cont_1": [50, 200, 200],
                        "other": [1, 2, 3]})
    df_test = unbalanced_concat(
        df1=df1,
        df2=df2,
        id_discrete=["discr"],
        id_continuous=["cont_0", "cont_1"]
    )
    assert str(df_test) == str(pd.DataFrame({
        'discr': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'cont_0': [0.0, 0.0, 50.0, 50.0, 50.0, 100.0, 100.0, 100.0, 125.0, 125.0, 125.0, 150.0, 150.0, 150.0],
        'cont_1': [50.0, 50.0, 100.0, 100.0, 100.0, 125.0, 125.0, 125.0, 150.0, 150.0, 150.0, 200.0, 200.0, 200.0],
        'other': [np.nan, 1.0, np.nan, 2.0, 3.0, np.nan, 2.0, 3.0, np.nan, 2.0, 3.0, np.nan, 2.0, 3.0]
    })), "\n" + str(df_test)


def test_unbalanced_concat_case7():
    # row 2 count_0: df1 < df2, row 2 count_1: df1 > df2,
    df1 = pd.DataFrame({"discr": [1, 1, 1],
                        "cont_0": [0, 50, 150],
                        "cont_1": [50, 200, 200]})
    df2 = pd.DataFrame({"discr": [1, 1, 1, 1],
                        "cont_0": [0, 100, 100, 150],
                        "cont_1": [100, 150, 150, 200],
                        "other": [1, 2, 3, 4]})
    df_test = unbalanced_concat(
        df1=df1,
        df2=df2,
        id_discrete=["discr"],
        id_continuous=["cont_0", "cont_1"]
    )
    assert str(df_test) == str(pd.DataFrame({
        'discr': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'cont_0': [0.0, 0.0, 50.0, 50.0, 100.0, 100.0, 100.0, 125.0, 125.0, 125.0, 150.0, 150.0],
        'cont_1': [50.0, 50.0, 100.0, 100.0, 125.0, 125.0, 125.0, 150.0, 150.0, 150.0, 200.0, 200.0],
        'other': [np.nan, 1.0, np.nan, 1.0, np.nan, 2.0, 3.0, np.nan, 2.0, 3.0, np.nan, 4.0]
    })), "\n" + str(df_test)


def test_unbalanced_concat_case8():
    # row 2 in df 2 overlaps entirely row 2 and row 3 in df 1
    df1 = pd.DataFrame({"discr": [1, 1, 1, 1],
                        "cont_0": [0, 50, 100, 150],
                        "cont_1": [50, 100, 150, 200]})
    df2 = pd.DataFrame({"discr": [1, 1, 1, 1],
                        "cont_0": [0, 20, 20, 180],
                        "cont_1": [20, 180, 180, 200],
                        "other": [1, 2, 3, 4]})
    df_test = unbalanced_concat(
        df1=df1,
        df2=df2,
        id_discrete=["discr"],
        id_continuous=["cont_0", "cont_1"]
    )
    assert str(df_test) == str(pd.DataFrame({
        'discr': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1],
        'cont_0': [0.0, 0.0, 20.0, 20.0, 20.0, 25.0, 25.0, 25.0, 47.0, 47.0, 47.0, 50.0, 50.0, 50.0, 73.0, 73.0, 73.0,
                   75.0, 75.0, 75.0, 100.0, 100.0, 100.0, 125.0, 125.0, 125.0, 127.0, 127.0, 127.0, 150.0, 150.0, 150.0,
                   153.0, 153.0, 153.0, 175.0, 175.0, 175.0, 180.0, 180.0],
        'cont_1': [20.0, 20.0, 25.0, 25.0, 25.0, 47.0, 47.0, 47.0, 50.0, 50.0, 50.0, 73.0, 73.0, 73.0, 75.0, 75.0,
                   75.0, 100.0, 100.0, 100.0, 125.0, 125.0, 125.0, 127.0, 127.0, 127.0, 150.0, 150.0, 150.0, 153.0,
                   153.0, 153.0, 175.0, 175.0, 175.0, 180.0, 180.0, 180.0, 200.0, 200.0],
        'other': [np.nan, 1.0, np.nan, 2.0, 3.0, np.nan, 2.0, 3.0, np.nan, 2.0, 3.0, np.nan, 2.0, 3.0, np.nan,
                  2.0, 3.0, np.nan, 2.0, 3.0, np.nan, 2.0, 3.0, np.nan, 2.0, 3.0, np.nan, 2.0, 3.0, np.nan,
                  2.0, 3.0, np.nan, 2.0, 3.0, np.nan, 2.0, 3.0, np.nan, 4.0]
    })), "\n" + str(df_test)


def test_unbalanced_concat_case9():
    # row 2 in df 1 overlaps entirely rows 2, 3 and 4 in df 2
    df1 = pd.DataFrame({"discr": [1, 1, 1],
                        "cont_0": [0, 20, 180],
                        "cont_1": [20, 180, 200]})
    df2 = pd.DataFrame({"discr": [1, 1, 1, 1, 1],
                        "cont_0": [0, 50, 100, 100, 150],
                        "cont_1": [50, 100, 150, 150, 200],
                        "other": [1, 2, 3, 4, 5]})
    df_test = unbalanced_concat(
        df1=df1,
        df2=df2,
        id_discrete=["discr"],
        id_continuous=["cont_0", "cont_1"]
    )
    assert str(df_test) == str(pd.DataFrame({
        'discr': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'cont_0': [0.0, 0.0, 20.0, 20.0, 25.0, 25.0, 47.0, 47.0, 50.0, 50.0, 73.0, 73.0, 75.0, 75.0, 100.0, 100.0,
                   100.0, 125.0, 125.0, 125.0, 127.0, 127.0, 127.0, 150.0, 150.0, 153.0, 153.0, 175.0, 175.0,
                   180.0, 180.0],
        'cont_1': [20.0, 20.0, 25.0, 25.0, 47.0, 47.0, 50.0, 50.0, 73.0, 73.0, 75.0, 75.0, 100.0, 100.0, 125.0,
                   125.0, 125.0, 127.0, 127.0, 127.0, 150.0, 150.0, 150.0, 153.0, 153.0, 175.0, 175.0, 180.0,
                   180.0, 200.0, 200.0],
        'other': [np.nan, 1.0, np.nan, 1.0, np.nan, 1.0, np.nan, 1.0, np.nan, 2.0, np.nan, 2.0, np.nan, 2.0, np.nan,
                  3.0, 4.0, np.nan, 3.0, 4.0, np.nan, 3.0, 4.0, np.nan, 5.0, np.nan, 5.0, np.nan, 5.0, np.nan, 5.0]
    })), "\n" + str(df_test)


def test_unbalanced_concat_case10():
    df1 = pd.DataFrame({"discr1": [1000, 1000, 1000, 1000, 1000, 2000, 2000, 2000],
                        "discr2": [1] * 2 + [2] * 3 + [1] * 3,
                        "cont1": [50, 100, 50, 100, 150, 50, 100, 150],
                        "cont2": [100, 150, 100, 150, 200, 100, 150, 200]})
    df2 = pd.DataFrame({"discr1": [1000, 2000, 2000, 2000],
                        "discr2": [1, 2, 2, 2],
                        "cont1": [30, 100, 100, 100],
                        "cont2": [110, 150, 150, 150],
                        "date": [2013, 2015, 2016, 2017]})
    df_test = unbalanced_concat(
        df1=df1,
        df2=df2,
        id_discrete=["discr1", "discr2"],
        id_continuous=["cont1", "cont2"]
    )
    assert str(df_test) == str(pd.DataFrame(
        {'discr1': [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 2000, 2000, 2000, 2000, 2000, 2000],
         'discr2': [1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2],
         'cont1': [30.0, 50.0, 50.0, 100.0, 100.0, 110.0, 50.0, 100.0, 150.0, 50.0, 100.0, 150.0, 100.0, 100.0, 100.0],
         'cont2': [50.0, 100.0, 100.0, 110.0, 110.0, 150.0, 100.0, 150.0, 200.0, 100.0, 150.0, 200.0, 150.0, 150.0,
                   150.0],
         'date': [2013.0, np.nan, 2013.0, np.nan, 2013.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                  2015.0, 2016.0, 2017.0]}
    )), "\n" + str(df_test)


def test_unbalanced_concat_case11():
    df1 = pd.DataFrame({"discr1": [1000, 1000],
                        "discr2": [1] * 2,
                        "cont1": [50, 85],
                        "cont2": [85, 150]})
    df2 = pd.DataFrame({"discr1": [1000],
                        "discr2": [1],
                        "cont1": [30],
                        "cont2": [120],
                        "date": [2013]})
    df_test = unbalanced_concat(
        df1=df1,
        df2=df2,
        id_discrete=["discr1", "discr2"],
        id_continuous=["cont1", "cont2"], verbose=True
    )
    assert str(df_test) == str(pd.DataFrame({
        'discr1': [1000] * 8,
        'discr2': [1] * 8,
        'cont1': [30.0, 50.0, 50.0, 75.0, 75.0, 85.0, 85.0, 120.0],
        'cont2': [50.0, 75.0, 75.0, 85.0, 85.0, 120.0, 120.0, 150.0],
        'date': [2013.0, np.nan, 2013.0, np.nan, 2013.0, np.nan, 2013.0, np.nan]
    })), "\n" + str(df_test)


def test_unbalanced_concat_case12():
    df1 = pd.DataFrame({"discr1": [1000, 1000],
                        "discr2": [1] * 2,
                        "cont1": [50, 80],
                        "cont2": [80, 150]})
    df2 = pd.DataFrame({"discr1": [1000],
                        "discr2": [1],
                        "cont1": [30],
                        "cont2": [120],
                        "date": [2013]})
    df_test = unbalanced_concat(
        df1=df1,
        df2=df2,
        id_discrete=["discr1", "discr2"],
        id_continuous=["cont1", "cont2"]
    )
    assert str(df_test) == str(pd.DataFrame({
        'discr1': [1000] * 10,
        'discr2': [1] * 10,
        'cont1': [30.0, 50.0, 50.0, 75.0, 75.0, 80.0, 80.0, 115.0, 115.0, 120.0],
        'cont2': [50.0, 75.0, 75.0, 80.0, 80.0, 115.0, 115.0, 120.0, 120.0, 150.0],
        'date': [2013.0, np.nan, 2013.0, np.nan, 2013.0, np.nan, 2013.0, np.nan, 2013.0, np.nan]
    })), "\n" + str(df_test)
