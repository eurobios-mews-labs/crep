# Copyright 2023 Eurobios
# Licensed under the CeCILL License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://cecill.info/
import warnings
from typing import Any, Tuple, Literal, Iterable, Dict, Optional, Union, Set

import numpy as np
import pandas as pd

from crep import tools
from crep.tools import concretize_aggregation


def merge(
        data_left: pd.DataFrame,
        data_right: pd.DataFrame,
        id_continuous: [Any, Any],
        id_discrete: iter,
        how: str,
        remove_duplicates: bool = False,
        verbose=False) -> pd.DataFrame:
    """
    This function aims at creating merge data frame

    Parameters
    ----------

    data_left
        data frame with continuous representation
    data_right
        data frame with continuous representation
    id_continuous
        iterable of length two that delimits the edges of the segment
    id_discrete: iterable
        iterable that lists all the columns on which to perform a classic merge
    how: str
        how to make the merge, possible options are

        - 'left'
        - 'right'
        - 'inner'
        - 'outer'

    remove_duplicates
        whether to remove duplicates
    verbose
    """
    __check_args_merge(data_left, data_right,
                       id_continuous, id_discrete, how)

    data_left = data_left.__deepcopy__()
    data_right = data_right.__deepcopy__()
    id_continuous = list(id_continuous)
    id_discrete = list(id_discrete)

    id_discrete_left = [col for col in data_left.columns if col in id_discrete]
    id_discrete_right = [col for col in data_right.columns if col in id_discrete]
    data_left, data_right = __fix_discrete_index(
        data_left, data_right,
        id_discrete_left,
        id_discrete_right)
    data_left.index = range(len(data_left))
    data_right.index = range(len(data_right))
    df_merge = __merge_index(data_left, data_right,
                             id_discrete=id_discrete,
                             id_continuous=id_continuous)
    df = pd.merge(
        df_merge,
        data_left[list(set(data_left.columns).difference(df_merge.columns))],
        left_on="left_idx", right_index=True, how="left")
    df = pd.merge(
        df,
        data_right[list(set(data_right.columns).difference(df_merge.columns))],
        left_on="right_idx", right_index=True, how="left")

    if how == "left":
        df = df.loc[df["left_idx"] != -1]
    if how == "right":
        df = df.loc[df["right_idx"] != -1]
    if how == "inner":
        df = df.loc[(df["right_idx"] != -1) & (df["left_idx"] != -1)]

    df = df.drop(["left_idx", "right_idx"], axis=1)
    df.index = range(len(df))
    if remove_duplicates:
        df = suppress_duplicates(df, id_discrete=id_discrete,
                                 id_continuous=id_continuous)
    if verbose:
        print("[merge] nb rows left  table frame ", data_left.shape[0])
        print("[merge] nb rows right table frame ", data_right.shape[0])
        print("[merge] nb rows outer table frame ", df.shape[0])
    return df


def unbalanced_merge(
        data_admissible: pd.DataFrame,
        data_not_admissible: pd.DataFrame,
        id_discrete: Iterable,
        id_continuous: [Any, Any], how) -> pd.DataFrame:
    """
    Merge admissible and non-admissible dataframes based on discrete and continuous identifiers.

    Parameters
    ----------

    data_admissible : pd.DataFrame
        DataFrame containing admissible data.
    data_not_admissible : pd.DataFrame
        DataFrame containing non-admissible data.
    id_discrete : list
        List of column names representing discrete identifiers.
    id_continuous : list
        List of column names representing continuous identifiers.
    how: str
        how to make the merge, possible options are

        - 'left'
        - 'right'
        - 'inner'
        - 'outer'

    Returns
    -------
    pd.DataFrame
        A DataFrame resulting from the unbalanced merge of admissible and non-admissible data.

    Notes
    -----
    The function performs the following steps:
    1. Combines and sorts the admissible and non-admissible data based on the identifiers.
    2. Resolves overlaps and conflicts between the admissible and non-admissible data.
    3. Merges and returns the final DataFrame.
    """

    if tools.admissible_dataframe(data_not_admissible, id_discrete, id_continuous):
        return merge(data_admissible, data_not_admissible,
                     id_discrete=id_discrete, id_continuous=id_continuous,
                     how=how)
    df_to_resolve_admissible = tools.build_admissible_data(data_not_admissible, id_discrete,
                                                           id_continuous)

    df_to_resolve_no_duplicates = df_to_resolve_admissible.drop_duplicates(subset=[*id_discrete, *id_continuous]).copy()
    df_to_resolve_no_duplicates["___id"] = range(len(df_to_resolve_no_duplicates))
    df_to_resolve_admissible = pd.merge(df_to_resolve_no_duplicates[[*id_discrete, *id_continuous, "___id"]],
                                        df_to_resolve_admissible,
                                        on=[*id_discrete, *id_continuous])


    df_ret = merge(
        df_to_resolve_no_duplicates[[*id_discrete, *id_continuous, "___id"]],
        data_admissible, id_discrete=id_discrete, id_continuous=id_continuous,
        how=how)

    df_ret = pd.merge(
        df_ret,
        df_to_resolve_admissible.drop(columns=[*id_discrete, *id_continuous]),
        on=["___id"], how=how, suffixes=("", "___")
    ).drop(columns="___id")
    return df_ret


def unbalanced_concat(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        id_discrete: Iterable[Any],
        id_continuous: [Any, Any],
        ignore_homogenize: bool = False,
        verbose: bool = False
) -> pd.DataFrame:
    """
    Concatenates the rows from two dataframes, and adjusts the lengths of the segments so that for each segment in the
    first dataframe there is a segment in the second dataframes with the same id_continuous characteristics, and
    vice versa. This function can handle duplicated rows in each other of the df, but not non-duplicated overlap.

    Parameters
    ----------
    df1 : pandas dataframe
    df2 : pandas dataframe
    id_discrete : list
        discrete columns (object or categorical)
    id_continuous : list of 2 column names
        continuous columns that delimit the segments' start and end
    ignore_homogenize : optional. boolean
        if True, ignore the homogenization function
    verbose: optional. boolean
        whether to print shape of df and if df is admissible at the end of the function.

    Returns
    -------
    df:  pandas dataframe
    """
    # ensures that ratio of segment size between the dataframes does not exceed 2
    df1_admissible = tools.admissible_dataframe(data=df1, id_discrete=id_discrete, id_continuous=id_continuous)
    df2_admissible = tools.admissible_dataframe(data=df2, id_discrete=id_discrete, id_continuous=id_continuous)

    if df1_admissible & df2_admissible:
        warnings.warn("Both dataframes are admissible. Consider using crep.merge().")

    if not ignore_homogenize:
        df1, df2 = homogenize_between(
            df1=df1,
            df2=df2,
            id_discrete=id_discrete,
            id_continuous=id_continuous
        )

    r1 = (df2[id_continuous[1]] - df2[id_continuous[0]]).max() / (df1[id_continuous[1]] - df1[id_continuous[0]]).min()
    r2 = (df1[id_continuous[1]] - df1[id_continuous[0]]).max() / (df2[id_continuous[1]] - df2[id_continuous[0]]).min()
    if r1 > 2 or r2 > 2:
        raise Exception("unbalanced_concat needs dataframes with segments size on a similar scale. "
                        "Ratios of segments sizes between dataframes should not exceed 2.")

    # assert tools.admissible_dataframe(data_admissible, id_discrete, id_continuous)
    df_idx_w = df1.copy()
    df_idx_s = df2.copy()
    df_idx_w["__t__"] = True
    df_idx_s["__t__"] = False

    df_idx = pd.concat((df_idx_w, df_idx_s))
    df_idx = df_idx.sort_values(
        [*id_discrete, id_continuous[1], "__t__"],
        ascending=[*[True] * len(id_discrete), True, False],
    ).reset_index(drop=True)

    # __new_seg__: bool, verifies whether the next line has the same id_discrete
    mask = df_idx[id_discrete].eq(df_idx[id_discrete].shift(1))
    df_idx["__new_seg__"] = ~((~mask).sum(axis=1) == 0)

    df_idx["__id1__"] = df_idx[id_continuous[0]]
    df_idx["__id2__"] = df_idx[id_continuous[1]]
    df_idx["__id3__"] = df_idx[id_continuous[0]]
    df_idx["__id1__"] = df_idx[id_continuous[0]].shift()
    df_idx["__id2__"] = df_idx[id_continuous[1]].shift()
    df_idx["__id3__"] = df_idx[id_continuous[0]].shift(-1)
    df_idx.loc[df_idx["__new_seg__"], ["__id1__", "__id2__"]] = -1
    df_idx.loc[df_idx["__new_seg__"].shift(-1).fillna(True), "__id3__"] = -1

    mask = (~df_idx[[*id_discrete, *id_continuous]].eq(df_idx[[*id_discrete, *id_continuous]].shift())).sum(axis=1)
    df_idx.loc[mask == 0, ["__id1__", "__id2__"]] = np.nan
    df_idx.loc[(mask == 0).shift(-1).fillna(False), "__id3__"] = np.nan

    df_idx[["__id1__", "__id2__"]] = df_idx[["__id1__", "__id2__"]].ffill()
    df_idx["__id3__"] = df_idx[["__id3__"]].bfill()
    df_idx[["__id1__", "__id2__", "__id3__"]] = df_idx[["__id1__", "__id2__", "__id3__"]].replace(-1, np.nan)

    # segments out: discontinuity between line n and line n+1
    c_out = pd.Series(
        df_idx["__new_seg__"]
        | (df_idx["__id2__"] < df_idx[id_continuous[0]])
        | (df_idx[id_continuous[1]] < df_idx["__id2__"])  # impossible because of sort_values by id_continuous[1]
        | (df_idx[id_continuous[0]] == df_idx["__id2__"])
    )

    c1_sup_id2 = df_idx[id_continuous[1]] > df_idx["__id2__"]

    #   |------------|
    #   |--------------------|
    # segments line n-1 and line n start at the same point, but segment line n-1 ends before segment line n
    c_resolve_1 = pd.Series((~c_out) & (df_idx[id_continuous[0]] == df_idx["__id1__"]) & c1_sup_id2)

    #   |--------------------|
    #          |--------------------|
    # segments line n-1 starts before line n and segment line n-1 ends before segment line n
    c_resolve_2 = pd.Series((~c_out) & (df_idx[id_continuous[0]] > df_idx["__id1__"]) & c1_sup_id2)

    #   |--------------------|
    #          |-------------|
    # segment line n-1 starts before segment line n and segments line n-1 and line n end at the same point
    c_resolve_3 = pd.Series(
        (~c_out) & (df_idx[id_continuous[0]] > df_idx["__id1__"]) & (~c1_sup_id2) & ~df_idx["__id3__"].isna())

    #        |----------|
    #   |--------------------|
    # segments line n-1 starts after line n and segment line n-1 ends before segment line n
    c_resolve_4 = pd.Series((~c_out) & (df_idx[id_continuous[0]] < df_idx["__id1__"]) & c1_sup_id2)

    #         |--------------|
    #   |--------------------|
    # segments line n-1 starts after line n and segments line n-1 and line n end at the same point
    c_resolve_5 = pd.Series((~c_out) & (df_idx[id_continuous[0]] < df_idx["__id1__"]) & (~c1_sup_id2))
    # neither of the above
    c_uncov = ~(c_out | c_resolve_1 | c_resolve_2 | c_resolve_3.shift(-1).fillna(False) | c_resolve_4 | c_resolve_5)

    created_columns = ["__new_seg__", "__id1__", "__id2__", "__id3__"]

    # =================
    # Resolve all
    list_df_new = []

    # c_resolve_1
    #   |------------|                   =>   |------------|
    #   |--------------------|           =>   |------------|-------|
    df_new = df_idx.loc[c_resolve_1, :].copy()
    df_temp = df_new.copy()
    df_new[id_continuous[0]] = df_new["__id2__"]
    df_temp[id_continuous[1]] = df_temp["__id2__"]
    df_new = pd.concat([df_new, df_temp]).drop(created_columns, axis=1)
    list_df_new.append(df_new)

    # c_resolve_2
    #   |--------------------|           =>   |------|-------------|
    #          |--------------------|    =>          |-------------|------|
    df_new = df_idx.loc[c_resolve_2, :].copy()
    df_temp = df_new.copy()
    mask = c_resolve_2.shift(-1).fillna(False) & (~df_idx["__id3__"].isna()) & (~c_resolve_4)
    df_temp2 = df_idx.loc[mask, :].copy()
    df_temp3 = df_idx.loc[mask, :].copy()
    df_new[id_continuous[0]] = df_new["__id2__"]
    df_temp[id_continuous[1]] = df_temp["__id2__"]
    df_temp2[id_continuous[0]] = df_temp2["__id3__"]
    df_temp3[id_continuous[1]] = df_temp2["__id3__"]
    df_new = pd.concat([df_new, df_temp, df_temp2, df_temp3]).drop(created_columns, axis=1)
    list_df_new.append(df_new)

    # c_resolve_3
    #   |--------------------|          =>   |------|-------------|
    #          |-------------|          =>          |-------------|
    mask = c_resolve_3.shift(-1).fillna(False)  # & ~c_resolve_4
    df_new = df_idx.loc[mask, :].copy()
    df_temp = df_new.copy()
    df_new[id_continuous[0]] = df_new["__id3__"]
    df_temp[id_continuous[1]] = df_temp["__id3__"]
    df_new = pd.concat([df_new, df_temp]).drop(created_columns, axis=1)
    list_df_new.append(df_new)

    # c_resolve_4
    #        |----------|              =>        |----------|
    #   |--------------------|         =>   |----|----------|----|
    df_new = df_idx.loc[c_resolve_4, :].copy()
    df_temp = df_new.copy()
    df_temp2 = df_new.copy()
    df_new[id_continuous[1]] = df_new["__id1__"]
    df_temp[id_continuous[0]] = df_temp["__id1__"]
    df_temp[id_continuous[1]] = df_temp["__id2__"]
    df_temp2[id_continuous[0]] = df_temp2["__id2__"]
    df_new = pd.concat([df_new, df_temp, df_temp2]).drop(created_columns, axis=1)
    list_df_new.append(df_new)

    # c_resolve_5
    #         |--------------|        =>         |--------------|
    #   |--------------------|        =>   |-----|--------------|
    df_new = df_idx.loc[c_resolve_5, :].copy()
    df_temp = df_new.copy()
    df_new[id_continuous[0]] = df_new["__id1__"]
    df_temp[id_continuous[1]] = df_temp["__id1__"]
    df_new = pd.concat([df_new, df_temp]).drop(created_columns, axis=1)
    list_df_new.append(df_new)

    # c_out corrections (see test_unbalanced_concat_case11)
    c_out[c_out & (c_resolve_2.shift(-1) | c_resolve_3.shift(-1))] = False

    df_res = pd.concat(
        list_df_new + [df_idx.loc[c_out | c_uncov, :].drop(created_columns, axis=1)]
    ).drop_duplicates(
    ).sort_values(
        [*id_discrete, id_continuous[1], "__t__", ],
        ascending=[*[True] * len(id_discrete), True, False]
    ).drop(
        "__t__", axis=1
    ).reset_index(drop=True)

    if verbose:
        print("post unbalanced_concat. Admissible:",
              tools.admissible_dataframe(data=df_res, id_discrete=id_discrete, id_continuous=id_continuous))
        print(df_res.shape)

    return df_res


def aggregate_constant(df: pd.DataFrame,
                       id_discrete: Iterable[Any],
                       id_continuous: [Any, Any],
                       ):
    """
    Parameters
    ----------
    df
    id_discrete
    id_continuous

    Returns
    -------
    """
    data_ = df.copy(deep=True)
    dtypes = data_.dtypes
    data_ = data_.sort_values([*id_discrete, *id_continuous])
    # 1/ detect unnecessary segment
    indexes = [*id_discrete, *id_continuous]
    no_index = list(set(data_.columns).difference(indexes))
    id1, id2 = id_continuous

    disc = tools.compute_discontinuity(data_, id_discrete, id_continuous)
    identical = False * np.ones_like(disc)

    index = data_.index

    data_1 = data_.loc[index[:-1], no_index].fillna(np.nan).values
    data_2 = data_.loc[index[1:], no_index].fillna(np.nan).values

    np_bool: np.array = np.equal(data_1, data_2)

    res = pd.Series(np_bool.sum(axis=1), index=index[:-1])
    res = pd.Series(res == len(no_index)).values

    identical[:-1] = res
    identical[:-1] = identical[:-1] & ~disc[1:]

    n = identical.sum()
    if n == 0:
        return df
    dat = pd.DataFrame(dict(
        identical=identical,
        keep=False * np.ones_like(disc)),
        index=data_.index)

    keep = list(set(np.where(identical)[0]).union(np.where(identical)[0] + 1))
    dat.loc[dat.index[keep], "keep"] = True

    data_merge = data_.sort_values([*id_discrete, *id_continuous])
    data_merge[f"{id1}_new"] = np.nan
    b = ~ dat["identical"]
    b_disc = [True] + list(~dat["identical"].values[:-1])
    data_merge.loc[b, f"{id2}_new"] = data_merge.loc[b, id2]
    data_merge.loc[b_disc, f"{id1}_new"] = data_merge.loc[b_disc, id1]

    data_merge[f"{id2}_new"] = data_merge[f"{id2}_new"].bfill()
    data_merge[f"{id1}_new"] = data_merge[f"{id1}_new"].ffill()

    data_merge = data_merge.drop(list(id_continuous), axis=1)
    data_merge = data_merge.rename({f"{id1}_new": id1, f"{id2}_new": id2},
                                   axis=1)
    return data_merge[df.columns].drop_duplicates().astype(dtypes)


def __merge_index(data_left,
                  data_right,
                  id_discrete,
                  id_continuous,
                  names=("left", "right")):
    id_ = [*id_discrete, *id_continuous]
    id_c = id_continuous

    data_left = data_left.loc[:, id_].dropna()
    data_right = data_right.loc[:, id_].dropna()
    df_merge = __merge(data_left, data_right,
                       id_discrete=id_discrete, id_continuous=id_c)
    return df_merge


def merge_event(
        data_left: pd.DataFrame,
        data_right: pd.DataFrame,
        id_discrete: Iterable[Any],
        id_continuous: [Any, Any],
        id_event
):
    """
    Assigns the details of events occurring at a specific points, in data_right, to the corresponding segment
    in data_left.

    Parameters
    ----------
    data_left : pd.DataFrame
        The left dataframe to be merged.
    data_right : pd.DataFrame
        The right dataframe to be merged.
    id_discrete : iterable
        The list of column names representing discrete identifiers for sorting and merging
        (e.g., categorical variables)
    id_continuous : list of two elements (Any, Any)
        A list with two elements representing the continuous index (e.g., time or numerical variables).
        The first element is the column name of the continuous identifier used for sorting.
    id_event:
        the name of the column containing the exact localisation of the event

    Returns
    -------
    pd.DataFrame
        A merged dataframe that combines `data_left` and `data_right`.

    """
    if not tools.admissible_dataframe(data=data_left, id_discrete=id_discrete, id_continuous=id_continuous):
        raise Exception("The left dataframe is not admissible. Consider using aggregate_duplicates() and "
                        "tools.build_admissible_data() if you want to make the dataframe admissible.")
    else:
        df1 = data_left.copy()
        df2 = data_right.copy()
        df1 = df1.fillna(1234.56789)
        df1["__t__"] = True
        df2["__t__"] = False
        df = pd.concat([df1, df2], axis=0)
        df.loc[df["__t__"], id_event] = df.loc[df["__t__"], id_continuous[1]]
        df = df.sort_values(by=[*id_discrete, id_event]).reset_index(drop=True)

        # ========== identify match with pk event in df2 and concerned row in df1 ==========
        mask = (~df[id_discrete].eq(df[id_discrete].shift()))
        df["__new_seg__"] = mask.sum(axis=1) > 0
        df["__new_seg_b_"] = np.nan
        df.loc[df["__t__"], "__new_seg_b_"] = df.loc[df["__t__"], "__new_seg__"]
        df["__new_seg_b_"] = df["__new_seg_b_"].bfill()
        df["__no_match__"] = False
        df.loc[(~df["__t__"]) & df["__new_seg_b_"], "__no_match__"] = True

        if df["__no_match__"].sum() > 1:
            warnings.warn("Not all events in data_right could be associated to a segment in date_left.")
            print(f"In merge_event, dropped: {df['__no_match__'].sum()}/{df2.shape[0]} rows")
        df = df.loc[~df["__no_match__"], :].reset_index(drop=True)

        # =========================== merge info from df1 and df2 ===========================
        df.loc[~df["__t__"], id_continuous] = np.nan
        df[df1.columns] = df[df1.columns].bfill()
        df.loc[df["__t__"], id_event] = np.nan
        df = df.replace(1234.56789, np.nan)

        df = df.sort_values(
            by=[*id_discrete, id_continuous[1], "__t__"],
            ascending=[*[True] * len(id_discrete), True, False]
        ).reset_index(drop=True)
        df = df.drop(columns=[col for col in df.columns if "__" in col])

        return df


def create_regular_segmentation(
        data: pd.DataFrame, length,
        id_discrete: iter,
        id_continuous: [Any, Any]
) -> pd.DataFrame:
    """
    Creates a regular segmentation of a DataFrame based on specified discrete and continuous columns,
    segmenting continuous values into equal-length intervals.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing the data to be segmented.
    length : int
        Length of each segment in the continuous interval. If set to 0, returns the original DataFrame.
    id_discrete : iterable
        Column names to be used as discrete identifiers for grouping data.
    id_continuous : list of Any, Any
        A list containing two continuous column names, where the first item represents the start value
        and the second item represents the end value for the segmentation.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with regular segments created based on the specified interval length and grouping columns.
        The resulting DataFrame includes columns for the discrete identifiers, the continuous start and end values
        for each segment.

    Notes
    -----
    The function calculates the number of segments for each discrete group by dividing the interval between the
    maximum and minimum continuous values by the given segment length. Segments are then evenly spaced within
    this interval for each discrete group.


    Raises
    ------
    ValueError
        If `length` is not a positive integer.
    """
    if length == 0:
        return data
    # For each couple we compute the number of segment given the length
    df_disc_f = data.groupby(id_discrete)[id_continuous[1]].max().reset_index()
    df_disc_d = data.groupby(id_discrete)[id_continuous[0]].min().reset_index()
    df_disc = pd.merge(df_disc_d, df_disc_f, on=id_discrete)

    df_disc["nb_coupon"] = np.round((df_disc[id_continuous[1]] - df_disc[id_continuous[0]]) / length).astype(int)
    df_disc["nb_coupon_cumsum"] = df_disc["nb_coupon"].cumsum()
    df_disc["nb_coupon_cumsum0"] = 0
    df_disc.loc[df_disc.index[1:], "nb_coupon_cumsum0"] = df_disc["nb_coupon_cumsum"].values[:-1]

    # Create empty regular segment table and we fill it with regular segment
    df_new = pd.DataFrame(index=range(df_disc["nb_coupon"].sum()),
                          columns=[*id_discrete, *id_continuous])
    for ix in df_disc.index:
        nb_cs = df_disc.loc[ix]
        value_temp = np.linspace(
            nb_cs[id_continuous[0]],
            nb_cs[id_continuous[1]],
            num=nb_cs['nb_coupon'] + 1,
            dtype=int)
        df_temp = pd.DataFrame(columns=[*id_discrete, *id_continuous])
        df_temp[id_continuous[0]] = value_temp[:-1]
        df_temp[id_continuous[1]] = value_temp[1:]
        df_temp[id_discrete] = nb_cs[id_discrete].values
        df_new.iloc[nb_cs["nb_coupon_cumsum0"]:nb_cs["nb_coupon_cumsum"]] = df_temp

    df_new["__id__"] = range(len(df_new))

    df_keep = merge(df_new, data,
                    id_continuous=id_continuous,
                    id_discrete=id_discrete,
                    how="left")

    df_new = df_new[df_new["__id__"].isin(df_keep["__id__"])]
    return df_new[[*id_discrete, *id_continuous]]


def __merge(df_left: pd.DataFrame, df_right: pd.DataFrame,
            id_discrete: Iterable[Any],
            id_continuous: [Any, Any],
            names=("left", "right")):
    index = [*id_discrete, *id_continuous]

    df_id1, df_id2, index_left, index_right = __refactor_data(
        df_left,
        df_right, id_continuous, id_discrete,
        names=names)
    df_id1_stretched = tools.create_continuity(
        df_id1, id_discrete=id_discrete,
        id_continuous=id_continuous)
    df_id2_stretched = tools.create_continuity(
        df_id2, id_discrete=id_discrete,
        id_continuous=id_continuous)

    df_id1_stretched.loc[df_id1_stretched[index_left].isna(), index_left] = -1
    df_id2_stretched.loc[df_id2_stretched[index_right].isna(), index_right] = -1

    df = pd.concat((df_id1_stretched, df_id2_stretched), sort=False)
    df = df.sort_values(by=index)

    id1, id2 = id_continuous
    df_merge = __table_jumps(df, *id_continuous, id_discrete)

    df_merge = df_merge.dropna()

    df_merge = pd.merge(
        df_merge,
        df_id1_stretched[[index_left, id1, *id_discrete]],
        on=[id1, *id_discrete], how="left")

    df_merge = pd.merge(
        df_merge,
        df_id2_stretched[[index_right, id1, *id_discrete]],
        on=[id1, *id_discrete], how="left")

    df_end1 = df_id1_stretched[[index_left, id2, *id_discrete]].rename(
        {index_left: index_left + "_end"}, axis=1)
    df_end2 = df_id2_stretched[[index_right, id2, *id_discrete]].rename(
        {index_right: index_right + "_end"}, axis=1)

    df_merge = pd.merge(
        df_merge,
        df_end1,
        left_on=[id1, *id_discrete],
        right_on=[id2, *id_discrete], how="left", suffixes=("", "_1"))
    df_merge = pd.merge(
        df_merge,
        df_end2,
        left_on=[id1, *id_discrete],
        right_on=[id2, *id_discrete], how="left", suffixes=("", "_2"))
    df_merge = df_merge.drop([f"{id2}_1", f"{id2}_2"], axis=1)

    # Tackle the problem of ending pad when there is discontinuity
    idx1 = (df_merge[index_left + "_end"].infer_objects(copy=False).fillna(-1) >= 0).values
    is_na_condition = df_merge.loc[:, index_left].isna()
    df_merge.loc[idx1 & is_na_condition, index_left] = -1

    idx2 = (df_merge[index_right + "_end"].infer_objects(copy=False).fillna(-1) >= 0).values
    is_na_condition_2 = df_merge.loc[:, index_right].isna()
    df_merge.loc[idx2 & is_na_condition_2, index_right] = -1
    df_merge = df_merge.drop([index_right + "_end", index_left + "_end"],
                             axis=1)

    discontinuity = tools.compute_discontinuity(df_merge, id_discrete, id_continuous)
    df_merge.loc[discontinuity & df_merge[
        index_left].isna(), index_left] = -1
    df_merge.loc[discontinuity & df_merge[
        index_right].isna(), index_right] = -1

    df_merge = df_merge.infer_objects(copy=False).ffill().drop("___t", axis=1)

    df_merge[[index_right, index_left, id1, id2]] = df_merge[
        [index_right, index_left, id1, id2]].astype(float).fillna(
        -1).astype(int)

    df_merge = df_merge.loc[
        ~(df_merge[index_left] + df_merge[index_right] == -2)]
    return df_merge


def __fix_discrete_index(
        data_left: pd.DataFrame,
        data_right: pd.DataFrame,
        id_discrete_left: iter,
        id_discrete_right: iter):
    if len(id_discrete_left) < len(id_discrete_right):
        data_right, data_left = __fix_discrete_index(
            data_right, data_left,
            id_discrete_right, id_discrete_left, )
        return data_left, data_right

    df_id_left = data_left.loc[:, id_discrete_left].drop_duplicates()
    df_id_right = data_right.loc[:, id_discrete_right].drop_duplicates()

    id_inter = [id_ for id_ in id_discrete_right if id_ in id_discrete_left]
    id_inter = list(id_inter)
    if len(id_inter) > 0:
        df_id_right = pd.merge(df_id_left, df_id_right, on=id_inter)
        data_right = pd.merge(df_id_right, data_right, on=id_discrete_right, how="left")
    return data_left, data_right


def suppress_duplicates(df, id_discrete, id_continuous):
    df = df.sort_values([*id_discrete, *id_continuous])
    df_duplicated = df.drop([*id_discrete, *id_continuous], axis=1)
    mat_duplicated = pd.DataFrame(
        df_duplicated.iloc[1:].values == df_duplicated.iloc[
                                         :-1].values)
    id1 = id_continuous[0]
    id2 = id_continuous[1]
    index = mat_duplicated.sum(axis=1) == df_duplicated.shape[1]
    index = np.where(index)[0]
    df1 = df.iloc[index]
    df2 = df.iloc[index + 1]
    idx_replace = df1[id2].values == df2[id1].values
    idx_to_agg = index[idx_replace]
    i_loc = df1.columns.get_loc(id2)
    df.iloc[idx_to_agg, i_loc] = df.iloc[idx_to_agg + 1, i_loc].values
    df = df.drop(df.index[idx_to_agg + 1])
    return df


def _increasing_continuous_index(df: pd.DataFrame, id_continuous: [Any, Any]):
    id1 = id_continuous[0]
    id2 = id_continuous[1]
    df[f"{id1}_new"] = df.loc[:, [id1, id2]].min(axis=1)
    df[f"{id2}_new"] = df.loc[:, [id1, id2]].max(axis=1)

    df = df.drop([id1, id2], axis=1)
    df = df.rename({f"{id1}_new": id1, f"{id2}_new": id2}, axis=1)
    return df


def __refactor_data(data_left, data_right, id_continuous, id_discrete,
                    names=("left", "right")):
    index = [*id_discrete, *id_continuous]
    data_left = _increasing_continuous_index(data_left, id_continuous)
    data_right = _increasing_continuous_index(data_right, id_continuous)
    index_right = names[1] + "_idx"
    index_left = names[0] + "_idx"
    df_id1 = data_left[index].drop_duplicates()
    df_id2 = data_right[index].drop_duplicates()
    df_id1.index.name = index_left
    df_id2.index.name = index_right
    df_id1 = df_id1.reset_index()
    df_id2 = df_id2.reset_index()

    df_id1[index_right] = np.nan
    df_id2[index_left] = np.nan

    df_id1 = df_id1.sort_values(by=index)
    df_id2 = df_id2.sort_values(by=index)
    return df_id1, df_id2, index_left, index_right


def __check_args_merge(data_left, data_right,
                       id_continuous,
                       id_discrete,
                       how):
    for c in [*id_continuous, *id_discrete]:
        if not (c in data_left.columns or c in data_right.columns):
            raise ValueError(f"{c} is not in columns")
    if not len(id_continuous) == 2:
        raise ValueError("Only two continuous index is possible")
    if how not in ["left", "right", "inner", "outer"]:
        raise ValueError('How must be in "left", "right", "inner", "outer"')


def __table_jumps(data, id1, id2, id_discrete):
    df_unique_start = data[[id1, *id_discrete]].rename(
        {id1: "___t"}, axis=1).drop_duplicates()

    df_unique_end = data[[id2, *id_discrete]].rename(
        {id2: "___t"}, axis=1).drop_duplicates()

    ret = pd.concat((df_unique_end, df_unique_start),
                    sort=False).sort_values(
        by=[*id_discrete, "___t"]).drop_duplicates()
    if len(ret) == 0:
        return ret
    ret.index = range(ret.shape[0])
    ret[id1] = -1
    ret.iloc[:-1, -1] = ret["___t"].iloc[:-1].values
    ret[id2] = -1
    ret.iloc[:-1, -1] = ret["___t"].iloc[1:].values
    return ret


def aggregate_duplicates(
        df: pd.DataFrame,
        id_discrete: Iterable[Any],
        id_continuous: [Any, Any],
        dict_agg: Optional[Dict[str, Iterable[Any]]] = None,
        verbose: bool = False
):
    """
    Removes duplicated rows by aggregating them.
    TODO : assess

    Parameters
    ----------
    df : pandas dataframe
    id_discrete : list
        discrete columns (object or categorical)
    id_continuous : list of 2 column names
        continuous columns that delimit the segments' start and end
    dict_agg: dict, keys: agg operator, values: list of columns or None
        specify which aggregation operator to apply for which column. If None, default is mean for all columns.
        id_continuous and id_discrete columns don't need to be specified in the dictionary
    verbose: boolean
        whether to print shape of df and if df is admissible at the end of the function.


    Returns
    -------
    df: pandas dataframe
        without duplicated rows

    Raises
    ------
    Exception
        When the dataframe df passed in argument does not contain any duplicated row
    """
    df = df.copy()
    cumul_ = tools.cumul_length(df, id_continuous=id_continuous)

    # split duplicates and non-duplicates
    mask = df.duplicated(subset=[*id_discrete, *id_continuous], keep=False)
    df_dupl = df.loc[mask, :].sort_values(by=[*id_discrete, id_continuous[1]]).reset_index(drop=True)
    if df_dupl.shape[0] == 0:
        raise Exception("The dataframe does not contain duplicated rows.")
    df_no_dupl = df.loc[~mask, :].sort_values(by=[*id_discrete, id_continuous[1]]).reset_index(drop=True)

    # preparation for groupby & agg
    same_than_above = df_dupl[[*id_discrete, *id_continuous]].eq(df_dupl[[*id_discrete, *id_continuous]].shift())
    same_than_above = (~same_than_above).sum(axis=1) == 0
    df_dupl["__lim__"] = 1
    df_dupl.loc[same_than_above, "__lim__"] = 0
    df_dupl["__lim__"] = df_dupl["__lim__"].cumsum()

    # =============== groupby & agg of df_dupl ==================
    drop_cols = set()  # columns that should be removed at the end ot the process
    df_gr = []  # list of dataframes that will further be concatenated
    colnames = []  # names of new columns

    if dict_agg is None:
        warnings.warn("dict_agg not specified. Default aggregation operator set to 'mean' for all features.")
        columns = [col for col in df.columns if col not in ["__lim__", *id_discrete, *id_continuous]]
        dict_agg = {"mean": columns}

    # define id_continuous agg operators
    if "min" in dict_agg.keys():
        dict_agg["min"].append(id_continuous[0])
    else:
        dict_agg["min"] = [id_continuous[0]]
    if "max" in dict_agg.keys():
        dict_agg["max"].append(id_continuous[1])
    else:
        dict_agg["max"] = [id_continuous[1]]

    group_by = [*id_discrete, "__lim__"]
    dict_renaming = {}
    for i, items in enumerate(dict_agg.items()):
        k, v = items
        if k == "mode":
            data = df_dupl[group_by + v].groupby(by=group_by).agg(lambda x: ", ".join(x.mode().to_list()[0])).reset_index().drop(group_by, axis=1)
        else:
            data = df_dupl[group_by + v].groupby(by=group_by).agg(k).reset_index().drop(group_by, axis=1)
        df_gr.append(data)
        colnames += [f"{k}_" + col for col in v]
        for col in v:
            drop_cols.add(col)
            # renaming columns in df_no_dupl
            rk = list(dict_renaming.keys())
            if col not in id_continuous:
                if col in rk:
                    df_no_dupl[col + f"_{len(rk)}"] = df_no_dupl[col]
                    dict_renaming[col + f"_{len(rk)}"] = tools.name_simplifier([f"{k}_" + col])[0]
                else:
                    dict_renaming[col] = tools.name_simplifier([f"{k}_" + col])[0]

    # concatenation of all groupby dataframes
    df_gr = pd.concat(df_gr, axis=1)
    df_gr.columns = tools.name_simplifier(colnames)
    df_dupl = df_dupl.drop_duplicates(subset=group_by).reset_index(drop=True)
    df_dupl = df_dupl.drop(columns=list(drop_cols))
    df_dupl.columns = tools.name_simplifier(df_dupl.columns)
    df_dupl = pd.concat([df_dupl, df_gr], axis=1)
    # drop unnecessary columns (those that were processed in group_by)
    df_dupl = df_dupl.drop(
        columns=["__lim__"],
    ).rename(
        columns={f"min_{id_continuous[0]}": id_continuous[0], f"max_{id_continuous[1]}": id_continuous[1]}
    ).sort_values(
        by=[*id_discrete, id_continuous[1]]
    ).reset_index(drop=True)

    # =============== recombining df_dupl & df_no_dupl ==================
    df_no_dupl = df_no_dupl.rename(columns=dict_renaming)
    df = pd.concat([df_dupl, df_no_dupl], axis=0)
    df = df.sort_values(by=[*id_discrete, id_continuous[1]]).reset_index(drop=True)
    df = tools.reorder_columns(df=df, id_discrete=id_discrete, id_continuous=id_continuous)

    if verbose:
        print("post aggregate_duplicates. Admissible:",
              tools.admissible_dataframe(data=df, id_discrete=id_discrete, id_continuous=id_continuous))
        print(df.shape)
        c = tools.cumul_length(df, id_continuous=id_continuous)
        print("cumulative length post:", c, "diff pre-post:", cumul_ - c)

    return df


def aggregate_continuous_data(
        df: pd.DataFrame,
        id_discrete: Iterable[Any],
        id_continuous: [Any, Any],
        target_size: int,
        dict_agg: Optional[Dict[str, Iterable[Any]]] = None,
        verbose: bool = False
) -> pd.DataFrame:
    """
    Aggregate segments to uniformize the size of smaller segments.

    Parameters
    ----------
    df : pandas dataframe
        without duplicated rows or overlapping rows
    id_discrete : list
        discrete columns (object or categorical)
    id_continuous : list of 2 column names
        continuous columns that delimit the segments' start and end
    target_size: integer > 0
        targeted segment size
    dict_agg: optional. dict, keys: agg operator, values: list of columns or None,
        specify which aggregation operator to apply for which column. If None, default is mean for all columns.
        id_continuous, id_discrete and add_group_by columns don't need to be specified in the dictionary
    verbose: optional. boolean
        whether to print shape of df and if df is admissible at the end of the function.


    Returns
    -------
    df: pandas dataframe

    Raises
    ------
    Exception
        When the dataframe df passed in argument is not admissible i.e. it contains overlapping rows and or duplicates
    """
    if not tools.admissible_dataframe(data=df, id_discrete=id_discrete, id_continuous=id_continuous):
        raise Exception("The dataframe is not admissible. Consider using aggregate_duplicates() and "
                        "crep.tools.build_admissible_data() if you want to make the dataframe admissible.")
    df = df.copy()
    cumul_ = tools.cumul_length(df, id_continuous=id_continuous)

    df["__lim__"] = tools.clusterize(
        df=df,
        id_discrete=id_discrete,
        id_continuous=id_continuous,
        target_size=target_size,
    )
    df = tools.concretize_aggregation(
        df=df,
        id_discrete=id_discrete,
        id_continuous=id_continuous,
        dict_agg=dict_agg,
        add_group_by="__lim__"
    )
    df = df.drop("__lim__", axis=1)

    if verbose:
        print("post aggregate_continuous_data. Admissible:",
              tools.admissible_dataframe(data=df, id_discrete=id_discrete, id_continuous=id_continuous))
        print(df.shape)
        c = tools.cumul_length(df, id_continuous=id_continuous)
        print("cumulative length post:", c, "diff pre-post:", cumul_ - c)

    return df


def split_segment(
        df: pd.DataFrame,
        id_discrete: Iterable[Any],
        id_continuous: [Any, Any],
        target_size: int,
        columns_sum_aggregation: Iterable[str] = None,
        verbose: bool = False
) -> pd.DataFrame:
    """
    Uniformizes segment size by splitting them into shorter segments close to target size.

    Parameters
    ----------
    df : pandas dataframe
        without duplicated rows or overlapping rows
    id_discrete : list
        discrete columns (object or categorical)
    id_continuous : list of 2 column names
        continuous columns that delimit the segments' start and end
    target_size: integer > 0
        targeted segment size
    columns_sum_aggregation: Iterable[str], optional
        Default to empty list. Some columns may have to be summed over several segments when creating super segments.
        If so, splitting a row and assigning to each new row the same value as in the original non-split row may
        result in inflated sums later on. To counter that, the columns that should later be summed are specified in
        this list. The values are transformed into ratios relative to the segment size, then the row is split, and
        then an inverse transformation is done to reassign a non-ratio value.
    verbose: optional. boolean
        whether to print shape of df and if df is admissible at the end of the function.

    Returns
    -------
    df: pandas dataframe
    """
    df = df.copy()
    if columns_sum_aggregation is None:
        columns_sum_aggregation = []

    df["__n_cut__"] = tools.n_cut_finder(
        df=df,
        id_discrete=id_discrete,
        id_continuous=id_continuous,
        target_size=target_size,
        method="split"
    )
    df["__n_cut_dyn__"] = df["__n_cut__"]

    if "__diff__" not in df.columns:
        df["__diff__"] = df[id_continuous[1]] - df[id_continuous[0]]

    for col in columns_sum_aggregation:
        df[col] = df[col] / df["__diff__"]

    new_rows = []
    while df["__n_cut_dyn__"].max() > 0:
        df_temp = df.loc[df["__n_cut_dyn__"] >= 1, :].copy()
        df_temp[id_continuous[1]] = (
                df_temp[id_continuous[0]]
                + df_temp["__diff__"] * ((df_temp["__n_cut_dyn__"]) / df_temp["__n_cut__"])
        ).round().astype("int")
        df_temp[id_continuous[0]] = (
                df_temp[id_continuous[0]]
                + df_temp["__diff__"] * ((df_temp["__n_cut_dyn__"] - 1) / df_temp["__n_cut__"])
        ).round().astype("int")
        new_rows.append(df_temp)
        df["__n_cut_dyn__"] -= 1
    df = pd.concat(new_rows, axis=0).sort_values(by=[*id_discrete, id_continuous[1]]).reset_index(drop=True)

    df["__diff__"] = df[id_continuous[1]] - df[id_continuous[0]]
    for col in columns_sum_aggregation:
        df[col] = df[col] * df["__diff__"]

    df = df.drop(["__diff__", "__n_cut__", "__n_cut_dyn__"], axis=1)

    if verbose:
        print("post split_segment. Admissible:",
              tools.admissible_dataframe(data=df, id_discrete=id_discrete, id_continuous=id_continuous))
        print(df.shape)

    return df


def homogenize_within(
        df: pd.DataFrame,
        id_discrete: Iterable[Any],
        id_continuous: [Any, Any],
        target_size: Optional[Union[float,int]] = None,
        method: Optional[Union[Literal["agg", "split"], Iterable[Literal["agg", "split"]], Set[Literal["agg", "split"]]]] = None,
        dict_agg: Optional[Dict[str, Iterable[Any]]] = None,
        strict_size: bool = False,
        verbose: bool = False
) -> pd.DataFrame:
    """
    Uniformizes segment size by splitting them into shorter segments close to target size. The uniformization aims
    to get a close a possible to target_size with +- 1.33 * target_size as maximum error margin.

    Parameters
    ----------
    df : pandas dataframe
        without duplicated rows or overlapping rows
    id_discrete : list
        discrete columns (object or categorical)
    id_continuous : list of 2 column names
        continuous columns that delimit the segments' start and end
    target_size: optional, integer > 0 or None
        targeted segment size. If None, the median is selected.
    method : optional str, either "agg" or "split"
        Whether to homogenize segment length by splitting long segments ("split") or by aggregating short segments ("agg") or both.
        Default to None lets the function define the method.
    strict_size: whether to strictly respect target_size specified in argument, if any specified.
        The function can change the target size if the value is not congruent with the method
    dict_agg: optional. dict, keys: agg operator, values: list of columns or None,
        specify which aggregation operator to apply for which column. If None, default is mean for all columns.
        id_continuous, id_discrete and add_group_by columns don't need to be specified in the dictionary
    verbose: optional. boolean
        whether to print shape of df and if df is admissible at the end of the function.

    Raises
    ------
    Exception:
        If method is not defined and if the function failed to select automatically a method.

    Returns
    -------
    df: pandas dataframe
    """
    # ==================
    # verify is method is applicable, and sets methods and target_size
    df = df.copy()

    if method is None:
        method = set()
    else:
        if type(method) is str:
            method = {method}
        else:
            method = set(method)
    df["__diff__"] = (df[id_continuous[1]] - df[id_continuous[0]])
    min_thresh = int(df["__diff__"].min() * 1.33)
    # method "agg" is not applicable if some rows are duplicated or if dict_agg is missing
    agg_applicable = (
            tools.admissible_dataframe(data=df, id_discrete=id_discrete, id_continuous=id_continuous)
            & ((dict_agg is not None) | ("agg" in method))
    )
    if not agg_applicable:
        warnings.warn("Method 'agg' is not applicable. The dataframe might either be non-admissible, or dict_agg is "
                      "not specified and 'agg' method was not specified either.")

    if len(method) == 0:
        if df["__diff__"].min() < target_size / 1.5 and agg_applicable:
            method.add("agg")
        elif df["__diff__"].max() > target_size * 1.33:
            method.add("split")
        else:
            warnings.warn("No method selected. Please, check whether the dataframe is admissible and "
                          "whether the target size is coherent given the size of the segments in the dataframe.")

    if target_size is None:
        target_size = int(df["__diff__"].median())
        warnings.warn(f"Unspecified target size set at median: {target_size}")

    if "agg" not in method and target_size > min_thresh and not strict_size:
        initial_ts = f"{target_size}"
        target_size = max(int(df["__diff__"].min() * 1.33), 20)
        warnings.warn(f"Specified target_size for method {method} was not congruent with segment sizes in the"
                      " dataframe. target_size has been modified from " + initial_ts + f" to {target_size}.")

    if "__diff__" in df.columns:
        df = df.drop("__diff__", axis=1)

    # ==================
    # apply method(s)
    col_sum_agg = []
    if dict_agg is not None:
        if "sum" in dict_agg.keys():
            col_sum_agg = dict_agg["sum"]

    if "split" in method or ("agg" in method and target_size < min_thresh):
        df = split_segment(
            df=df,
            id_discrete=id_discrete,
            id_continuous=id_continuous,
            target_size=target_size // 3 if "agg" in method else target_size,
            columns_sum_aggregation=col_sum_agg,
            verbose=verbose
        )

    if "agg" in method:
        df = aggregate_continuous_data(
            df=df,
            id_discrete=id_discrete,
            id_continuous=id_continuous,
            target_size=target_size,
            dict_agg=dict_agg,
            verbose=verbose
        )

    return df


def homogenize_between(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        id_discrete: Iterable[Any],
        id_continuous: Iterable[Any],
        dict_agg_df1: Optional[Dict[str, Iterable[str]]] = None,
        dict_agg_df2: Optional[Dict[str, Iterable[str]]] = None,
        keep_df1: bool = False,
        verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    If the ratio of max segment size in one dataframe and min segment size in the other dataframe > 2, it may create
    issues in the unbalanced_concat function. homogenize_between changes the segments sizes in the dataframes to
    target a ratio < 2 between the dataframes.

    Demonstration of the problem:
    Example of the early merging phase in unbalanced_merge:
    row 1: from df1 30 50
    row 2: from df1 50 70
    row 3, from df2 15 85
    unbalanced_merge will detect the overlap between row 2 and row 3, but it will not detect that row 1 is
    also overlapping with row 3. Thus, created splits will be 15-50, 50-70, 70-85 instead of being
    15-30, 30-50, 50-70, 70-85. Ratio of max segment in df2 / min segment in df1 < 2 eliminates this problem:
    row 1: from df2 15-45
    row 2, from df1 30-50
    row 3: from df1 50 70
    row 4: from df2 45-85
    => splits will be 15-30, 30-45, 45-50, 50-70, 70-85

    Parameters
    ----------
    df1 : pandas dataframe
    df2 : pandas dataframe
    id_discrete : list
        discrete columns (object or categorical)
    id_continuous : list of 2 column names
        continuous columns that delimit the segments' start and end
    dict_agg_df1: optional, Dict[str, Iterable[str]]
        dictionary with settings about how to handle the columns in df1 that are neither id_discrete nor id_continuous
    dict_agg_df2: optional, Dict[str, Iterable[str]]
        dictionary with settings about how to handle the columns in df2 that are neither id_discrete nor id_continuous
    keep_df1: optional, bool
        default to False. If True, the segmentation in df1 does not change. Only df2 adapts to df1.
    verbose: optional. boolean
        whether to print shape of df and if df is admissible at the end of the function.

    Returns
    -------
    df: pandas dataframe

    """
    df1 = df1.copy()
    df2 = df2.copy()

    df1["__diff__"] = (df1[id_continuous[1]] - df1[id_continuous[0]])
    df2["__diff__"] = (df2[id_continuous[1]] - df2[id_continuous[0]])

    min_diff = df1["__diff__"].min()
    min_diff_ref = df2["__diff__"].min()
    if 1.33 * min_diff < min_diff_ref:
        target_size = int(1.33 * min_diff)
    else:
        target_size = int(1.33 * min_diff_ref)
    print(f"homogenize_between: chosen target size: {target_size}")

    df2 = homogenize_within(
        df=df2.drop("__diff__", axis=1),
        id_discrete=id_discrete,
        id_continuous=id_continuous,
        target_size=target_size,
        dict_agg=dict_agg_df2,
        verbose=verbose
    )

    if not keep_df1:
        df1 = homogenize_within(
            df=df1.drop("__diff__", axis=1),
            id_discrete=id_discrete,
            id_continuous=id_continuous,
            target_size=target_size,
            dict_agg=dict_agg_df1,
            verbose=verbose
        )

    return df1, df2


def segmentation_irregular(
        df: pd.DataFrame,
        id_discrete: Iterable[Any],
        id_continuous: [Any, Any],
        length_target,
        length_minimal,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df: pd.DataFrame
    id_discrete: Iterable[str]
        list of name of columns of categorical type
    id_continuous: [str, str]
        list of name of 2 columns of numerical type, indicating the start and the end of the segment
    length_target
        length to obtain at the end of the segmentation
    length_minimal
        When there are gaps in the dataframe, define the length beyond which this could be considered as a
        deliberate break in the segmentation and not as missing data. Under this threshold, a new row will
        be created to ensure the continuity between successive segments in the dataframe.

    Returns
    -------
    pd.DataFrame
        New dataframe containing only the columns id_discrete and id_continuous, with the length of the segments
        adjusted to be as close as possible to length_target.
    """

    df_new = tools.create_continuity(
        df=df,
        id_discrete=id_discrete,
        id_continuous=id_continuous,
        limit=length_minimal,
        sort=False
    )

    df_new = homogenize_within(
        df=df_new[[*id_discrete, *id_continuous]],
        id_discrete=id_discrete,
        id_continuous=id_continuous,
        method=["agg", "split"],
        target_size=length_target,
        dict_agg=None,
        strict_size=False,
        verbose=False
    )

    return df_new


def segmentation_regular(
        df: pd.DataFrame,
        id_discrete: Iterable[Any],
        id_continuous: [Any, Any],
        length_target,
        length_gap_filling,
) -> pd.DataFrame:
    data = tools.create_continuity(
        df.__deepcopy__(),
        id_discrete=id_discrete,
        id_continuous=id_continuous,
        limit=length_gap_filling)
    indexes = [*id_continuous, *id_discrete]

    df_disc_f = data.groupby(id_discrete)[id_continuous[1]].max().reset_index()
    df_disc_d = data.groupby(id_discrete)[id_continuous[0]].min().reset_index()
    df_disc = pd.merge(df_disc_d, df_disc_f, on=id_discrete)
    df_disc["nb_coupon"] = np.round((df_disc[id_continuous[1]] - df_disc[id_continuous[0]]) / length_target).astype(int)
    df_disc["nb_coupon_cumsum"] = df_disc["nb_coupon"].cumsum()
    df_disc["nb_coupon_cumsum0"] = 0
    df_disc.loc[df_disc.index[1:], "nb_coupon_cumsum0"] = df_disc["nb_coupon_cumsum"].values[:-1]

    # Create empty regular segment table and we fill it with regular segment
    df_new = pd.DataFrame(index=range(df_disc["nb_coupon"].sum()),
                          columns=id_discrete)

    accumulator = {k: [] for k in indexes}

    for ix in df_disc.index:
        nb_cs = df_disc.loc[ix].to_dict()
        value_temp = np.linspace(
            nb_cs[id_continuous[0]],
            nb_cs[id_continuous[1]],
            num=int(nb_cs["nb_coupon"] + 1), dtype=int)

        accumulator[id_continuous[0]] += list(value_temp[:-1])
        accumulator[id_continuous[1]] += list(value_temp[1:])
        for idd in id_discrete:
            accumulator[idd] += list(np.array([nb_cs[idd]] * len(value_temp[1:])))

    for index in indexes:
        df_new[index] = accumulator[index]

    df_new.index = range(len(df_new))

    return df_new


def aggregate_on_segmentation(
        df_segmentation: pd.DataFrame,
        df_data: pd.DataFrame,
        id_discrete: Iterable[str],
        id_continuous: Iterable[str],
        dict_agg: Optional[Dict[str, Iterable[str]]]  = None
):
    """
    adds data to segmentation

    Parameters
    ----------
    df_segmentation: pd.DataFrame
        the dataframe containing the segmentation. Should contain only columns id_discrete and id_continuous
    df_data: pd.DataFrame
        the dataframe containing the features to fit to the segmentation. Should contain the columns
        id_discrete and id_continuous as well as other columns for the features of interest.
    id_discrete
    id_continuous
    dict_agg:

    Returns
    -------
    pd.DataFrame:
        a dataframe with the feature data fitted to the new segmentation.
    """
    # verification of requirements
    for col in id_continuous + id_discrete:
        if col not in df_segmentation.columns or col not in df_data.columns:
            raise Exception(f"Error: {col} is not present in both dataframes df_segm and df_feat.")

    is_df_segm_admissible = tools.admissible_dataframe(
        data=df_segmentation, id_discrete=id_discrete, id_continuous=id_continuous
    )
    is_df_feat_admissible = tools.admissible_dataframe(
        data=df_data, id_discrete=id_discrete, id_continuous=id_continuous
    )
    if not is_df_segm_admissible or not is_df_feat_admissible:
        raise Exception("Error: Both dataframes should be admissible:"
                        f"Is df_segm admissible? {is_df_segm_admissible}"
                        f"Is df_feat admissible? {is_df_feat_admissible}")

    # homogenize_between() reduces the difference in segment size between df_feat and df_segm. More precisely, it
    # adjusts df_feat to df_segm. This may reduce the risk of error when using merge().
    df_segmentation, df_features = homogenize_between(
        df1=df_segmentation,
        df2=df_data,
        id_discrete=id_discrete,
        id_continuous=id_continuous,
        dict_agg_df1=None,
        dict_agg_df2=dict_agg,
        keep_df1=True,
        verbose=False
    )

    df_segmentation["__id__"] = 1
    df_segmentation["__id__"] = df_segmentation["__id__"].cumsum()

    # merging the segmentations in both df
    df_merge = merge(
        data_left=df_segmentation,
        data_right=df_features,
        id_continuous=id_continuous,
        id_discrete=id_discrete,
        how="left",
        remove_duplicates=False,
        verbose=False
    )

    # groupby based on the settings in dict_agg and based on grouping variable __id__
    df_merge = concretize_aggregation(
        df=df_merge,
        id_discrete=id_discrete,
        id_continuous=id_continuous,
        dict_agg=dict_agg,
        add_group_by="__id__",
        verbose=False
    )

    df_merge = df_merge.drop(columns=["__id__"])

    df_merge = tools.reorder_columns(df_merge, id_discrete, id_continuous)

    return df_merge



