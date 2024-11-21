# Copyright 2023 Eurobios
# Licensed under the CeCILL License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://cecill.info/

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crep import merge, unbalanced_merge, unbalanced_concat, aggregate_constant
from crep import tools

plt.style.use('./examples/.matplotlibrc')


def plot(data: pd.DataFrame, continuous_index, data_col, color="C1",
         legend=None):
    id1, id2 = continuous_index
    for i, r in data.iterrows():
        label = legend if i == 0 else None
        plt.plot([r[id1], r[id2]], [r[data_col], r[data_col]], marker=".",
                 color=color, label=label)


def illustrate_2_inputs(
        function: callable,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        parameters: dict,
        id_discrete,
        continuous_index,
        data_columns=None,

):
    df_left = tools.sort(df_left, id_discrete=id_discrete, id_continuous=continuous_index)
    df_right = tools.sort(df_right, id_discrete=id_discrete, id_continuous=continuous_index)
    if data_columns is None:
        df_left["data left"] = np.array(range(len(df_left))) / len(df_left)
        df_right["data right"] = np.array(range(len(df_right))) / len(df_right) + 0.5
        data_columns = ["data left", "data right"]

    ret = function(df_left, df_right,
                   id_continuous=continuous_index,
                   id_discrete=id_discrete, **parameters)

    _, axes = plt.subplots(nrows=2, sharex=True)

    sel = ret[id_discrete].drop_duplicates().iloc[-1:]
    ret = pd.merge(ret, sel, how="right", on=id_discrete)
    df_left = df_left.merge(sel, how="right", on=id_discrete)
    df_right = df_right.merge(sel, how="right", on=id_discrete)

    ret = tools.sort(ret, id_discrete=id_discrete, id_continuous=continuous_index)
    ret[data_columns[0]] = ret[data_columns[0]] + (2 * np.mod(range(len(ret)), 2) - 1) * 0.005
    ret[data_columns[1]] = ret[data_columns[1]] + (2 * np.mod(range(len(ret)), 2) - 1) * 0.005
    plt.sca(axes[0])
    plot(df_left, continuous_index, data_col=data_columns[0], color="C1", legend='data left')
    plot(df_right, continuous_index, data_col=data_columns[1], color="C2", legend='data right')
    plt.ylabel("raw data")
    plt.legend()
    plt.sca(axes[1])
    plot(ret, continuous_index, data_col=data_columns[0], color="C1")
    plot(ret, continuous_index, data_col=data_columns[1], color="C2")
    plt.xlabel("$t$")
    plt.ylabel(f"{function.__name__} func.")
    plt.savefig(f"examples/{function.__name__}.png")


def illustrate_1_input(
        function: callable,
        df: pd.DataFrame,
        parameters: dict,
        id_discrete,
        continuous_index,
        data_columns=None,

):
    df = tools.sort(df, id_discrete=id_discrete, id_continuous=continuous_index)
    if data_columns is None:
        df["data"] = np.array(range(len(df))) / len(df)
        data_columns = ["data"]

    ret = function(df,
                   id_continuous=continuous_index,
                   id_discrete=id_discrete, **parameters)

    _, axes = plt.subplots(nrows=2, sharex=True)

    sel = ret[id_discrete].drop_duplicates().iloc[-1:]
    ret = pd.merge(ret, sel, how="right", on=id_discrete)
    df = pd.merge(df, sel, how="right", on=id_discrete)

    ret = tools.sort(ret, id_discrete=id_discrete, id_continuous=continuous_index)
    ret[data_columns[0]] = ret[data_columns[0]] + (2 * np.mod(range(len(ret)), 2) - 1) * 0.005
    plt.sca(axes[0])
    plot(df, continuous_index, data_col=data_columns[0], color="C1", legend='data input')
    plt.ylabel("raw data")
    plt.legend()
    plt.sca(axes[1])
    plot(ret, continuous_index, data_col=data_columns[0], color="C1")
    plt.xlabel("$t$")
    plt.ylabel(f"{function.__name__} func.")
    plt.savefig(f"examples/{function.__name__}.png")


plt.ioff()
illustrate_2_inputs(merge,
                    pd.read_csv("data/base_left.csv"),
                    pd.read_csv("data/base_right.csv"),
                    dict(how="outer", ),
                    id_discrete=["id"],
                    continuous_index=["t1", "t2"]
                    )
illustrate_2_inputs(unbalanced_merge,
                    pd.read_csv("data/base_left.csv"),
                    pd.read_csv("data/base_right.csv"),
                    dict(),
                    id_discrete=["id"],
                    continuous_index=["t1", "t2"]
                    )
illustrate_2_inputs(unbalanced_concat,
                    pd.read_csv("data/base_left.csv"),
                    pd.read_csv("data/base_right.csv"),
                    dict(),
                    id_discrete=["id"],
                    continuous_index=["t1", "t2"]
                    )

illustrate_1_input(tools.build_admissible_data,
                   pd.read_csv("data/base_left.csv"),
                   dict(),
                   id_discrete=["id"],
                   continuous_index=["t1", "t2"]
                   )

illustrate_1_input(tools.build_admissible_data,
                   pd.read_csv("data/base_left.csv"),
                   dict(),
                   id_discrete=["id"],
                   continuous_index=["t1", "t2"]
                   )

illustrate_1_input(function=aggregate_constant,
                   df=pd.read_csv("data/advanced_left.csv"),
                   parameters=dict(),
                   id_discrete=["id", "id2"],
                   continuous_index=["t1", "t2"],
                   data_columns=["data1"]
                   )
