# Copyright 2023 Eurobios
# Licensed under the CeCILL License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://cecill.info/

import pandas as pd
import numpy as np

from crep import tools

id_discrete, id_continuous = ["id", "id2"], ["t1", "t2"]

data = pd.read_csv("examples/data/data_zones.csv")


def test_no_overlapping(get_examples):
    assert sum(tools.get_overlapping(
        get_examples[0], ["id"], ["t1", "t2"]
    )) == 0
    assert tools.admissible_dataframe(get_examples[0], ["id"], id_continuous)


def test_overlapping(get_examples):
    df = get_examples[0]
    df.loc[1, "t1"] = 5
    ret = tools.get_overlapping(df, ["id"], id_continuous)
    assert sum(ret) == 2


def test_sample_overlapping(get_examples):
    df = get_examples[0]
    df.loc[1, "t1"] = 5
    ret = tools.sample_non_admissible_data(df, ["id"], id_continuous)
    assert ret.equals(df.loc[[0, 1]])


def test_create_zones():
    df = data.sort_values(by=[*id_discrete, *id_continuous])
    df_out = tools.create_zones(df.drop(columns="__zone__"), id_discrete=id_discrete, id_continuous=id_continuous)
    df_out = df_out.sort_values(by=[*id_discrete, *id_continuous])
    assert all(df_out["__zone__"].values == df["__zone__"].values)


def test_build_admissible_data_frame():
    ret = tools.build_admissible_data(df=data.drop(columns="__zone__"), id_discrete=id_discrete,
                                      id_continuous=id_continuous)
    ret = ret.drop_duplicates(subset=[*id_discrete, *id_continuous])

    assert tools.admissible_dataframe(ret, id_discrete=id_discrete, id_continuous=id_continuous)


def test_fix_point_non_admissible(get_examples):
    df_non_admissible = tools.sample_non_admissible_data(data, id_discrete, id_continuous)
    df_non_admissible_2 = tools.sample_non_admissible_data(df_non_admissible, id_discrete, id_continuous)

    assert df_non_admissible.equals(df_non_admissible_2)

    df_non_admissible = tools.sample_non_admissible_data(get_examples[0], ['id'], id_continuous)
    df_non_admissible_2 = tools.sample_non_admissible_data(df_non_admissible, ['id'], id_continuous)

    assert df_non_admissible.equals(df_non_admissible_2)

    df_non_admissible = tools.sample_non_admissible_data(get_examples[1], ['id'], id_continuous)
    df_non_admissible_2 = tools.sample_non_admissible_data(df_non_admissible, ['id'], id_continuous)

    assert df_non_admissible.equals(df_non_admissible_2)


def test_create_continuity(get_examples):
    df_left, df_right = get_examples
    df = tools.create_continuity(df_left, id_discrete=["id"], id_continuous=["t1", "t2"], sort=True)
    assert len(df_left) + 2 == len(df)
    assert df.isna().sum().sum() > 0
    df2 = tools.create_continuity(df, id_discrete=["id"], id_continuous=["t1", "t2"], sort=True)
    assert len(df2) == len(df)
    df3 = tools.create_continuity(df_left, id_discrete=["id"], id_continuous=["t1", "t2"], sort=True, limit=5)
    assert len(df_left) == len(df3)


def test_mark_new_segment():
    df = pd.DataFrame({"discr1": [1000, 1000, 1000, 1000, 1000, 2000, 2000, 2000],
                        "discr2": [1] * 2 + [2] * 3 + [1] * 3,
                        "cont1": [50, 100, 50, 100, 150, 50, 100, 150],
                        "cont2": [100, 150, 100, 150, 200, 100, 150, 200]})
    new_segm = tools.mark_new_segment(df, id_discrete=["discr1", "discr2"], id_continuous=["cont1", "cont2"])
    assert new_segm.to_list() == [True, False, True, False, False, True, False, False], "\n" + str(new_segm)


def test_cumul_segment_length():
    df = pd.DataFrame({"discr1": [1000, 1000, 1000, 1000, 1000, 2000, 2000, 2000],
                        "discr2": [1] * 2 + [2] * 3 + [1] * 3,
                        "cont1": [50, 100, 50, 100, 150, 50, 100, 150],
                        "cont2": [100, 150, 100, 150, 200, 100, 150, 200]})

    cumul = tools.cumul_segment_length(df, id_discrete=["discr1", "discr2"], id_continuous=["cont1", "cont2"])
    assert cumul.to_list() == [50, 100, 50, 100, 150, 50, 100, 150], "\n" + str(cumul)


def test_concretize_aggregation():
    df = pd.DataFrame({"discr1": [1000, 1000, 1000, 1000, 1000, 2000, 2000, 2000],
                        "discr2": [1] * 2 + [2] * 3 + [1] * 3,
                        "cont1": [50, 100, 50, 100, 150, 50, 100, 150],
                        "cont2": [100, 150, 100, 150, 200, 100, 150, 200],
                        "date": [2008, 2010, 2014, 2016, 2018, 2020, 2022, 2024]})
    df_test = tools.concretize_aggregation(
        df,
        id_discrete=["discr1", "discr2"],
        id_continuous=["cont1", "cont2"],
        dict_agg={"min": ["date"], "max": ["date"], "sum": ["date"], "mean": ['date']}, verbose=True
    )
    assert \
        ((df_test["min_date"].to_list() == [2008, 2014, 2020])
         & (df_test["max_date"].to_list() == [2010, 2018, 2024])
         & (df_test["sum_date"].to_list() == [4018, 6048, 6066])
         & (df_test["mean_date"].to_list() == [2009.0, 2016.0, 2022.0])), \
        "\n" + str(df_test)


def test_n_cut_finder_case1():
    df = pd.DataFrame({"discr1": [1000] * 8 + [2000] * 4,
                        "discr2": [1] * 4 + [2] * 4 + [1] * 4,
                        "cont1": [50, 100, 150, 200, 40, 75, 135, 178, 50, 90, 150, 210],
                        "cont2": [100, 150, 200, 250, 75, 135, 178, 211, 90, 150, 210, 280],
                        "date": list(range(2000, 2024, 2))})
    n_cut = tools.n_cut_finder(
        df=df,
        id_discrete=["discr1", "discr2"],
        id_continuous=["cont1", "cont2"],
        target_size=100,
        method="agg"
    )
    mask = n_cut.isna()
    assert list(n_cut.loc[mask].index) == [0, 1, 2, 4, 5, 6, 8, 9, 10], "\n" + str(n_cut)


def test_n_cut_finder_case2():
    df = pd.DataFrame({"discr1": [1000] * 5 + [2000] * 3,
                        "discr2": [1] * 2 + [2] * 3 + [1] * 3,
                        "cont1": [50, 100, 40, 75, 300, 50, 90, 250],
                        "cont2": [100, 150, 75, 300, 380, 90, 250, 310]})
    n_cut = tools.n_cut_finder(
        df=df,
        id_discrete=["discr1", "discr2"],
        id_continuous=["cont1", "cont2"],
        target_size=50,
        method="split"
    )
    assert n_cut.to_list() == [1, 1, 1, 5, 2, 1, 3, 1], "\n" + str(n_cut)


def test_clusterize_case1():
    df = pd.DataFrame({"discr1": [1000] * 8 + [2000] * 4,
                        "discr2": [1] * 4 + [2] * 4 + [1] * 4,
                        "cont1": [50, 100, 150, 200, 40, 75, 135, 178, 50, 90, 150, 210],
                        "cont2": [100, 150, 200, 250, 75, 135, 178, 211, 90, 150, 210, 280],
                        "date": list(range(2000, 2024, 2))})
    lim = tools.clusterize(
        df=df,
        id_discrete=["discr1", "discr2"],
        id_continuous=["cont1", "cont2"],
        target_size=140
    )
    assert lim.to_list() == [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5],  "\n" + str(lim)


def test_clusterize_case2():
    df = pd.DataFrame({"discr1": [830341] * 4,
                        "discr2": ["v2"] * 4,
                        "cont1": [637, 704, 634008, 634062],
                        "cont2": [704, 789, 634062, 634130],
                        "date": [2000, 2002, 2004, 2006]})
    lim = tools.clusterize(
        df=df,
        id_discrete=["discr1", "discr2"],
        id_continuous=["cont1", "cont2"],
        target_size=216
    )
    assert lim.to_list() == [1, 1, 2, 2], "\n" + str(lim)


def test_create_continuity_limit():
    df = pd.DataFrame({"discr1": [1000] * 4 + [2000] * 4,
                       "discr2": [1] * 2 + [2] * 2 + [1] * 2 + [2] * 2,
                       "cont1": [50, 80, 50, 90, 150, 190, 80, 1200],
                       "cont2": [80, 150, 85, 125, 172, 250, 105, 1235],
                       "date": list(range(2000, 2016, 2))})
    df_test = tools.create_continuity_modified(
        df=df,
        id_discrete=["discr1", "discr2"],
        id_continuous=["cont1", "cont2"],
        limit=100,
        sort=True
    )
    correct = str(pd.DataFrame({
        'discr1': [1000, 1000, 1000, 1000, 1000, 2000, 2000, 2000, 2000, 2000],
        'discr2': [1, 1, 2, 2, 2, 1, 1, 1, 2, 2],
        'cont1': [50, 80, 50, 85, 90, 150, 172, 190, 80, 1200],
        'cont2': [80, 150, 85, 90, 125, 172, 190, 250, 105, 1235],
        'date': [2000.0, 2002.0, 2004.0, np.nan, 2006.0, 2008.0, np.nan, 2010.0, 2012.0, 2014.0]}))
    print(correct)
    assert str(df_test) == correct, "\n" + str(df_test) + "\n" + correct


def test_name_simplifier():
    names = ["discr1", "discr2", "cont1", "cont2", "max_date", "mean_length", "min_depth", "sum_benefits", "max_mean_costs",
             "mean_mean_age"]
    new_names = tools.name_simplifier(names)
    assert (new_names == ['discr1', 'discr2', 'cont1', 'cont2', 'max_date', 'mean_length', 'min_depth', 'sum_benefits',
                          'max_costs', 'mean_age']
            ), new_names

