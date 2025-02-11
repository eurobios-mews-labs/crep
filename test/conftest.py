# Copyright 2023 Eurobios
# Licensed under the CeCILL License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://cecill.info/
import pandas as pd
from crep.table import DataFrameContinuous
from pytest import fixture


@fixture(scope="module")
def get_examples():
    df_left = pd.read_csv("data/base_left.csv")
    df_right = pd.read_csv("data/base_right.csv")
    return df_left, df_right


@fixture(scope="module")
def get_examples_dataframe_continuous():
    df_right = DataFrameContinuous(
        pd.read_csv("data/base_right.csv"),
        discrete_index=["id"],
        continuous_index=["t1", "t2"]
    )
    df_left = DataFrameContinuous(
        pd.read_csv("data/base_left.csv"),
        discrete_index=["id"],
        continuous_index=["t1", "t2"]
    )
    return df_left, df_right


@fixture(scope="module")
def get_advanced_examples():
    df_left = pd.read_csv("data/advanced_left.csv")
    df_right = pd.read_csv("data/advanced_right.csv")
    return df_left, df_right


@fixture(scope="module")
def get_advanced_examples_dataframe_continuous():
    df_left = DataFrameContinuous(
        pd.read_csv("data/advanced_left.csv"),
        discrete_index=["id"],
        continuous_index=["t1", "t2"]
    )
    df_right = DataFrameContinuous(
        pd.read_csv("data/advanced_right.csv"),
        discrete_index=["id"],
        continuous_index=["t1", "t2"]
    )
    return df_left, df_right
