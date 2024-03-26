# Copyright 2023 Eurobios
# Licensed under the CeCILL License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://cecill.info/

from crep.tools import get_overlapping, admissible_dataframe, sample_non_admissible_data

__args__ = ["id"], ["t1", "t2"]


def test_no_overlapping(get_examples):
    assert sum(get_overlapping(
        get_examples[0], ["id"], ["t1", "t2"]
    )) == 0
    assert admissible_dataframe(get_examples[0], *__args__)


def test_overlapping(get_examples):
    df = get_examples[0]
    df.loc[1, "t1"] = 5
    ret = get_overlapping(df, *__args__)
    assert sum(ret) == 1


def test_sample_overlapping(get_examples):
    df = get_examples[0]
    df.loc[1, "t1"] = 5
    ret = sample_non_admissible_data(df, *__args__)
    assert ret.equals(df.loc[[1]])
