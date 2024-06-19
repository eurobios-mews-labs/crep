# Copyright 2023 Eurobios
# Licensed under the CeCILL License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://cecill.info/
import pandas as pd
from pytest import fixture


@fixture(scope="module")
def get_examples():
    df_left = pd.DataFrame(
        dict(id=[1, 1, 2, 2, 2],
             t1=[0, 10, 0, 100, 120],
             t2=[10, 100, 90, 110, 130],
             data1=[0.2, 0.2, 0.1, 0.3, 0.2])
    )
    df_right = pd.DataFrame(
        dict(id=[1, 1, 2, 2, 2],
             t1=[5, 10, 0, 100, 120],
             t2=[10, 80, 90, 110, 130],
             data2=[0.2, 0.2, 0.1, 0.3, 0.2])
    )
    return df_left, df_right


@fixture(scope="module")
def get_advanced_examples():
    df_left = pd.DataFrame(
        dict(id=[1, 1, 2, 2, 2],
             id2=["a", "b", "b", "b", "b"],
             t1=[0, 10, 0, 100, 120],
             t2=[10, 100, 90, 110, 130],
             data1=[0.2, 0.2, 0.1, 0.3, 0.2])
    )
    df_right = pd.DataFrame(
        dict(id=[1, 2, 2, 2, 2],
             t1=[5, 10, 0, 100, 120],
             t2=[10, 80, 90, 110, 130],
             data2=[0.2, 0.2, 0.1, 0.3, 0.2])
    )
    return df_left, df_right
