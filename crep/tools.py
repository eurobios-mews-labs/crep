# Copyright 2023 Eurobios
# Licensed under the CeCILL License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://cecill.info/
import warnings
from typing import Any, Iterable, Literal, Dict, Optional, Union

import numpy as np
import pandas as pd


def build_admissible_data(
        df: pd.DataFrame,
        id_discrete: Iterable[Any],
        id_continuous: [Any, Any]
) -> pd.DataFrame:
    df.index = range(len(df.index))
    df_non_admissible = sample_non_admissible_data(df, id_discrete, id_continuous).__deepcopy__()
    df_non_admissible = create_zones(df_non_admissible, id_discrete, id_continuous)

    df_non_admissible["__id__"] = range(len(df_non_admissible))
    df_add_non_admissible = df_non_admissible[[*id_discrete, *id_continuous, "__id__", "__zone__"]]
    all_id_continuous = df_non_admissible[id_continuous[0]].to_list()
    all_id_continuous += df_non_admissible[id_continuous[1]].to_list()

    # create good segmentation
    df_ret = pd.concat([df_non_admissible[[*id_discrete, "__zone__"]]] * 2)
    df_ret[id_continuous[0]] = all_id_continuous
    df_ret["__disc__"] = compute_discontinuity(df_ret, id_discrete, id_continuous)
    df_ret = df_ret.sort_values(by=[*id_discrete, id_continuous[0]])
    df_ret[id_continuous[1]] = - df_ret[id_continuous[0]].diff(periods=-1) + df_ret[id_continuous[0]]
    df_ret = df_ret.dropna().drop(columns="__disc__")
    df_ret = df_ret.drop_duplicates().dropna()
    df_ret = df_ret[df_ret[id_continuous[1]] != df_ret[id_continuous[0]]]
    df_ret = df_ret.sort_values(by=[*id_discrete, id_continuous[0]])
    df_ret[id_continuous[1]] = df_ret[id_continuous[1]].astype(df[id_continuous[1]].dtype)

    df_ret = pd.merge(df_ret, df_add_non_admissible,
                      on=list(id_discrete) + ["__zone__"], suffixes=("", "_tmp"))
    id_continuous_tmp = [str(i) + "_tmp" for i in id_continuous]
    c = df_ret[id_continuous[0]] < df_ret[id_continuous_tmp[1]]
    c &= df_ret[id_continuous[1]] > df_ret[id_continuous_tmp[0]]
    df_ret = df_ret.loc[c].drop(columns=id_continuous_tmp)

    df_ret = pd.merge(df_ret, df_non_admissible.drop(columns=[*id_discrete, *id_continuous, "__zone__"]), on="__id__"
                      ).drop(columns=["__id__", "__zone__"])
    df_ret = df_ret.astype(df_non_admissible.dtypes.drop(["__id__", "__zone__"]))

    df_ret_all = df[~get_overlapping(df, id_discrete, id_continuous)]
    df_ret_all = pd.concat((df_ret_all, df_ret))
    df_ret_all = df_ret_all.sort_values(by=[*id_discrete, id_continuous[0]])
    df_ret_all.index = range(len(df_ret_all.index))
    return df_ret_all


def create_zones(
        df: pd.DataFrame,
        id_discrete: Iterable[Any],
        id_continuous: [Any, Any]
):
    """
    Create overlapping zone identifiers in the DataFrame based on discrete and continuous ID columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the df.
    id_discrete : iter
        An iterable of column names that are considered discrete identifiers.
    id_continuous : iter
        An iterable of column names that are considered continuous identifiers.

    Returns
    -------
    pd.DataFrame
        The DataFrame with an additional '__zone__' column indicating the zone for each row.

    Notes
    -----
    The function works by sorting the DataFrame based on the given discrete and continuous identifiers,
    and then creating a zone identifier (`__zone__`) that groups rows based on specific conditions.

    Steps:
    1. Sort the DataFrame based on discrete identifiers and the second continuous identifier.
    2. Assign a forward index (`__zf__`) based on the sorted order.
    3. Sort the DataFrame based on discrete identifiers and the first continuous identifier.
    4. Assign a backward index (`__zi__`) based on the sorted order.
    5. Determine zones where the forward and backward indices are equal (`c_zone`).
    6. Check if the start of a zone is greater than or equal to the end of the previous zone (`c_inner`).
    7. Identify changes in discrete identifiers (`c_disc`).
    8. Combine the conditions to create the final zone identifier (`__zone__`).

    Examples
    --------
    >>> df = {
    ...     'id': [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    ...     't1': [932, 996, 2395, 2395, 3033, 3628, 4126, 4140, 4154, 316263, 316263, 316471, 316471],
    ...     't2': [2395, 2324, 3033, 3628, 3035, 4140, 4140, 5508, 5354, 316399, 316471, 317406, 317557],
    ...     'LONGUEUR': [1463, 1328, 638, 1233, 2, 512, 14, 1368, 1200, 136, 208, 935, 1086],
    ...     '__zone__': [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4]
    ... }
    >>> df = pd.DataFrame(df)
    >>> create_zones(df, ['id'], ['t1', 't2'])
    """
    df_out = df.__deepcopy__()
    if "__zone__" in df.columns:
        df_out = df_out.drop(columns='__zone__')

    df_out = df_out.sort_values([*id_discrete, id_continuous[1]])
    df_out["__zf__"] = range(len(df_out))
    df_out = df_out.sort_values([*id_discrete, id_continuous[0]])
    df_out["__zi__"] = range(len(df_out))
    c_zone = (df_out["__zf__"] - df_out["__zi__"]) == 0

    df_out["__id2_prev__"] = df[id_continuous[1]]
    df_out.loc[df_out.index[1:], "__id2_prev__"] = df_out.loc[df_out.index[:-1], id_continuous[1]].values
    c_inner = df_out[id_continuous[0]] >= df_out["__id2_prev__"]

    c_disc = np.array(df_out[id_discrete].iloc[1:].values == df_out[id_discrete].iloc[:-1].values)
    c_disc = c_disc.mean(axis=1) == 1
    c_disc = ~ np.concatenate(([True], c_disc))

    df_out["__zone__"] = (c_zone & c_inner) | c_disc
    df_out["__zone__"] = df_out["__zone__"].cumsum()

    return df_out.loc[df.index, [*df.columns.to_list(), "__zone__"]]


def get_overlapping(df: pd.DataFrame,
                    id_discrete: Iterable[Any],
                    id_continuous: [Any, Any]
                    ) -> pd.Series:
    df = create_zones(df, id_discrete, id_continuous)
    overlap = df["__zone__"].duplicated(keep=False)
    return overlap


def admissible_dataframe(data: pd.DataFrame,
                         id_discrete: Iterable[Any],
                         id_continuous: [Any, Any]
                         ):
    return sum(get_overlapping(data, id_discrete,
                               id_continuous)) == 0


def sample_non_admissible_data(data: pd.DataFrame,
                               id_discrete: Iterable[Any],
                               id_continuous: [Any, Any]
                               ) -> pd.DataFrame:
    return data[get_overlapping(data, id_discrete,
                                id_continuous)]


def compute_discontinuity(
        df,
        id_discrete: Iterable[Any],
        id_continuous: [Any, Any]
):
    """
    Compute discontinuity in rail segment. The i-th element in return
    will be True if i-1 and i are discontinuous

    """
    discontinuity = np.zeros(len(df)).astype(bool)
    for col in id_discrete:
        if col in df.columns:
            discontinuity_temp = np.concatenate(
                ([False], df[col].values[1:] != df[col].values[:-1]))
            discontinuity |= discontinuity_temp

    if id_continuous[0] in df.columns and id_continuous[1] in df.columns:
        discontinuity_temp = np.concatenate(
            ([False], df[id_continuous[0]].values[1:] != df[id_continuous[
                1]].values[:-1]))
        discontinuity |= discontinuity_temp
    return discontinuity


def create_continuity(
        df: pd.DataFrame,
        id_discrete: Iterable[Any],
        id_continuous: [Any, Any],
        limit=None,
        sort=False
) -> pd.DataFrame:
    df_in = df.__deepcopy__()
    col_save = np.array(df_in.columns)
    index = [*id_discrete, *id_continuous]
    df_in["discontinuity"] = compute_discontinuity(df_in, id_discrete, id_continuous)
    if df_in["discontinuity"].sum() == 0:
        return df
    else:
        mask = (df[id_discrete].eq(df[id_discrete].shift())).sum(axis=1) < len(list(id_discrete))
        ix__ = np.where(df_in["discontinuity"].values & ~mask)[0]
        df_add = pd.DataFrame(columns=df_in.columns, index=range(len(ix__)))
        df_add[index] = df_in.iloc[ix__][index].values
        df_add[id_continuous[0]] = df_in.iloc[ix__ - 1].loc[:, id_continuous[1]].values
        df_add[id_continuous[1]] = df_in.iloc[ix__].loc[:, id_continuous[0]].values
        if limit is not None:
            df_add = df_add[(df_add[id_continuous[1]] - df_add[id_continuous[0]]) < limit]
        df_in = pd.concat((df_in, df_add.dropna(axis=1, how='all')), axis=0)
        df_in = df_in[df_in[id_continuous[0]] < df_in[id_continuous[1]]]
    if sort:
        df_in = df_in.sort_values([*id_discrete, *id_continuous])
    return df_in.loc[:, col_save].reset_index(drop=True)


def cumul_length(df: pd.DataFrame, id_continuous: [Any, Any]) -> int:
    """ Returns the sum of all segments sizes in the dataframe. """
    diff = df[id_continuous[1]] - df[id_continuous[0]]
    return diff.sum()


def reorder_columns(df: pd.DataFrame, id_discrete: Iterable[Any], id_continuous: [Any, Any]):
    other = [col for col in df.columns if col not in [*id_discrete, *id_continuous]]
    return df[[*id_discrete, *id_continuous, *other]]


def name_simplifier(names: Iterable[str]):
    list_agg_op = ["mean", "max", "min", "sum"]
    new_names = []
    for n in names:
        n = n.split("_")
        if n[0] in list_agg_op and n[1] in list_agg_op:
            del n[1]
            n = "_".join(n)
        else:
            n = "_".join(n)
        new_names.append(n)
    return new_names


def mark_new_segment(df: pd.DataFrame, id_discrete: Iterable[Any], id_continuous: [Any, Any]) -> pd.Series:
    """
    Creates a boolean pd.Series aligning with df indices. True: there is a change any of the id_discrete
    value between row n and row n-1 or there is a discontinuity (shown by id_continuous) between row n and row n-1
    Seems to be equivalent to crep.tools.compute_discontinuity

    Parameters
    ----------
    df : pandas dataframe
    id_discrete : list
        discrete columns (object or categorical)
    id_continuous : list of 2 column names
        continuous columns that delimit the segments' start and end

    Returns
    -------
    df: boolean pandas series
    """
    mask = df[id_discrete].eq(df[id_discrete].shift())
    new_segm = mask.sum(axis=1) < len(id_discrete)
    new_segm = new_segm | (~df[id_continuous[0]].eq(df[id_continuous[1]].shift()))
    return new_segm


def compute_cumulated_length(
        df: pd.DataFrame,
        id_discrete: Iterable[Any],
        id_continuous: [Any, Any]
) -> pd.Series:
    """
    TODO : compute_cumulated_length.
    Computes cumulative sum of segment length for each unique combination of id_discrete.

    Parameters
    ----------
    df : pandas dataframe
        without duplicated rows or overlapping rows
    id_discrete : list
        discrete columns (object or categorical)
    id_continuous : list of 2 column names
        continuous columns that delimit the segments' start and end

    Returns
    -------
    df: pandas series with integers
    """
    if not admissible_dataframe(data=df, id_discrete=id_discrete, id_continuous=id_continuous):
        raise Exception("The dataframe is not admissible. Consider using aggregate_duplicates() and "
                        "crep.tools.build_admissible_data() if you want to make the dataframe admissible.")
    df = df.copy()
    df["__new_seg__"] = mark_new_segment(df, id_discrete, id_continuous)
    df["__diff__"] = (df[id_continuous[1]] - df[id_continuous[0]])
    cumul = df["__diff__"].cumsum()
    df.loc[df["__new_seg__"], "__reset_cumul__"] = cumul.shift().loc[df["__new_seg__"]].fillna(0)
    cumul = cumul - df["__reset_cumul__"].ffill()
    # Correction for the fact that for the last row of each segment, __cumul__ == 0
    cumul.loc[cumul == 0] = cumul.shift() + df["__diff__"]
    return cumul


def concretize_aggregation(
        df: pd.DataFrame,
        id_discrete: Iterable[Any],
        id_continuous: [Any, Any],
        dict_agg: Optional[Dict[str, Iterable[Any]]],
        add_group_by: Optional[Union[Any, Iterable[Any]]] = None,
        verbose: bool = False
) -> pd.DataFrame:
    """
    Groupby + aggregation operations

    Parameters
    ----------
    df : pandas dataframe
        without duplicated rows or overlapping rows
    id_discrete : list
        discrete columns (object or categorical)
    id_continuous : list of 2 column names
        continuous columns that delimit the segments' start and end
    dict_agg: dict, keys: agg operator, values: list of columns or None,
        specify which aggregation operator to apply for which column. If None, default is mean for all columns.
        id_continuous, id_discrete and add_group_by columns don't need to be specified in the dictionary
    add_group_by : optional. column name or list of column names
        Additional columns to consider when grouping by
    verbose: boolean
        whether to print shape of df and if df is admissible at the end of the function.

    Returns
    -------
    df : pandas series with integers

    Raises
    ------
    Exception
        When the dataframe df passed in argument is not admissible i.e. it contains overlapping rows and or duplicates
    """
    if not admissible_dataframe(data=df, id_discrete=id_discrete, id_continuous=id_continuous):
        raise Exception("The dataframe is not admissible. Consider using aggregate_duplicates() and "
                        "crep.tools.build_admissible_data() if you want to make the dataframe admissible.")

    cumul_ = cumul_length(df, id_continuous=id_continuous)

    drop_cols = set()  # columns that should be removed at the end ot the process
    df_gr = []  # list of dataframes that will further be concatenated
    col_names = []  # names of new columns

    group_by = id_discrete
    if type(add_group_by) is str:
        group_by = group_by + [add_group_by]
    elif type(add_group_by) is list:
        group_by = group_by + add_group_by

    if dict_agg is None:
        warnings.warn("dict_agg not specified. Default aggregation operator set to 'mean' for all features.")
        columns = [col for col in df.columns if col not in [*group_by, *id_discrete, *id_continuous]]
        numerical_columns = list(df[columns].select_dtypes("number").columns)
        categorical_columns = list(df[columns].select_dtypes("object").columns)
        dict_agg = {}
        if len(numerical_columns) > 0:
            dict_agg = {"mean": numerical_columns}
        if len(categorical_columns) > 0:
            dict_agg["mode"] = categorical_columns

    # define id_continuous agg operators
    if "min" in dict_agg.keys():
        dict_agg["min"].append(id_continuous[0])
    else:
        dict_agg["min"] = [id_continuous[0]]
    if "max" in dict_agg.keys():
        dict_agg["max"].append(id_continuous[1])
    else:
        dict_agg["max"] = [id_continuous[1]]

    for i, items in enumerate(dict_agg.items()):
        k, v = items
        # Means are weighted by the length of the segments. Sums are not
        # To apply weights: mean = sum of (variable * length of segment) / sum of lengths of segments
        if k == "mode":
            data = df[group_by + v].groupby(by=group_by).agg(lambda x: ", ".join(x.mode().to_list())).reset_index().drop(group_by, axis=1)
            df_gr.append(data)
        elif k == "mean":
            # divider
            df["__diff__"] = df[id_continuous[1]] - df[id_continuous[0]]
            divider = pd.concat([df["__diff__"]] * len(v), axis=1)
            divider.columns = v
            divider = pd.concat([df[group_by], divider], axis=1)
            divider = divider.groupby(by=group_by).agg("sum").reset_index().drop(group_by, axis=1)
            # mean calculation
            df[v] = df[v].mul(df["__diff__"], axis=0)
            data = df[group_by + v].groupby(by=group_by).agg("sum").reset_index().drop(group_by, axis=1) / divider
            df_gr.append(data)
            drop_cols.add("__diff__")
        else:
            data = df[group_by + v].groupby(by=group_by).agg(k).reset_index().drop(group_by, axis=1)
            df_gr.append(data)
        col_names += [f"{k}_" + col for col in v]
        for col in v:
            drop_cols.add(col)

    # concatenation of all groupby dataframes
    df_gr = pd.concat(df_gr, axis=1)
    df_gr.columns = name_simplifier(col_names)
    df = df.drop(list(drop_cols), axis=1)
    df = df.drop_duplicates(group_by).reset_index(drop=True)
    df = pd.concat([df, df_gr], axis=1)
    # drop unnecessary columns (those that were processed in group_by)

    df = df.rename(columns={f"min_{id_continuous[0]}": id_continuous[0], f"max_{id_continuous[1]}": id_continuous[1]})

    if verbose:
        print("post concretize_agg. Admissible:",
              admissible_dataframe(data=df, id_discrete=id_discrete, id_continuous=id_continuous))
        print(df.shape)
        c = cumul_length(df, id_continuous=id_continuous)
        print("cumulative length post:", c, "diff pre-post:", cumul_ - c)

    return df


def n_cut_finder(
        df: pd.DataFrame,
        id_discrete: Iterable[Any],
        id_continuous: [Any, Any],
        target_size: int,
        method: Literal["agg", "split"]
) -> pd.Series:
    """
    Finds in how many sub-segments the segment should be cut (method = "split") or find where to stop the aggregation of
    segments into a super segment (method = "agg"). The returned value of the function is the pd.Series of the column
     __n_cut__

    If method is "agg", the __n_cut__ contains non-NaN value everywhere but in the last row before a change of
    id_discrete value. The non-NaN values represent how many super-segments should result from the aggregation of the
    previous rows with NaN values.

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
    method : str, either "agg" or "split"
        Whether to find n_cut for aggregating (agg) or for splitting (split)

    Returns
    -------
    df: pandas series
        agg: series with floats and NaN. Floats are displayed in the rows that mark new segments.
        The remaining rows contain NaN. The float values indicates the number of possible target_sizes divisions in the
        segment (the sum of the previous NaN rows)
        split: series with integers >= 1. They indicate in how many segments the current row should be divided.

    Raises
    ------
    Exception
        When the dataframe df passed in argument is not admissible i.e. it contains overlapping rows and or duplicates
    """
    created_columns = []
    df = df.copy()
    if method == "agg":
        if not admissible_dataframe(data=df, id_discrete=id_discrete, id_continuous=id_continuous):
            raise Exception("The dataframe is not admissible. Consider using aggregate_duplicates() and "
                            "crep.tools.build_admissible_data() if you want to make the dataframe admissible.")
        if "__new_seg__" not in df.columns:
            df["__new_seg__"] = mark_new_segment(df, id_discrete, id_continuous)
        if "__cumul__" not in df.columns:
            df["__cumul__"] = compute_cumulated_length(df, id_discrete, id_continuous).values
        df["__n_cut__"] = np.nan
        mask = df["__new_seg__"].shift(-1).fillna(True)
        df.loc[mask, "__n_cut__"] = (
            ((df["__cumul__"] + (target_size // 1.5)) / target_size)
            .loc[mask]
            .replace(0, 1)
        )
    else:
        df["__diff__"] = df[id_continuous[1]] - df[id_continuous[0]]
        created_columns.append("__diff__")
        df["__n_cut__"] = abs(df["__diff__"] - target_size // 2) // target_size + 1
    return df["__n_cut__"]


def clusterize(
        df: pd.DataFrame,
        id_discrete: Iterable[Any],
        id_continuous: [Any, Any],
        target_size: int,
) -> pd.Series:
    """
    TODO: create_cluster_by_size
    Defines where to limit segment aggregation when uniformizing segment size to target size.

    Parameters
    ----------
    df : pandas dataframe.
        The dataframe should be not have duplicated or overlapping rows.
    id_discrete : list
        discrete columns (object or categorical)
    id_continuous : list of 2 column names
        continuous columns that delimit the segments' start and end
    target_size: integer > 0
        targeted segment size

    Returns
    -------
    df : pandas series
        with common identifiers (integers) for the segments that should be grouped together.

    Raises
    ------
    Exception
        When the dataframe df passed in argument is not admissible i.e. it contains overlapping rows and or duplicates
    """
    if not admissible_dataframe(data=df, id_discrete=id_discrete, id_continuous=id_continuous):
        raise Exception("The dataframe is not admissible. Consider using aggregate_duplicates() and "
                        "crep.tools.build_admissible_data() if you want to make the dataframe admissible.")
    df = df.copy()
    df["__diff__"] = df[id_continuous[1]] - df[id_continuous[0]]
    if target_size < 2 * df["__diff__"].max():
        raise ValueError("target_size should at least be 2 times larger than the maximum segment length.")
    else:
        df["__new_seg__"] = mark_new_segment(df, id_discrete, id_continuous)
        df["__cumul__"] = compute_cumulated_length(df, id_discrete, id_continuous).values
        df = df.sort_values([*id_discrete, id_continuous[1]]).reset_index(drop=True)

        # how many cuts should be done based on __cumul__ (created by cumul_segment_length())
        df["__n_cut__"] = n_cut_finder(
            df=df,
            id_discrete=id_discrete,
            id_continuous=id_continuous,
            target_size=target_size,
            method="agg"
        )
        df["__n_cut_dyn__"] = df["__n_cut__"]

        df["__target__"] = (df["__cumul__"] // df["__n_cut__"].floordiv(1).replace(0, 1)).bfill()
        # modulos of target length indicate local minima = where to limit segment aggregation
        df["__%_a__"] = df["__cumul__"] % df["__target__"]
        df["__%_b__"] = df["__target__"] - (df["__cumul__"] % df["__target__"])
        df["__%_min__"] = df[["__%_a__", "__%_b__"]].min(axis=1)

        df["__lim__"] = 0
        mask = (
                (df["__%_min__"].diff()) <= 0
                & (df["__%_min__"].diff(-1) <= 0)
                & (~df["__new_seg__"])
        )
        df.loc[mask, "__lim__"] = -1
        df.loc[df["__lim__"].diff() == 1, "__lim__"] = 1
        df.loc[df["__new_seg__"], "__lim__"] = 1
        df["__lim__"] = df["__lim__"].replace(-1, 0)
        df["__lim__"] = df["__lim__"].cumsum().ffill().fillna(0)

        # the 1st row of a super segment (marked by __new_seg__) is necessarily the start of a new segment. Since
        # target_size > 2 * size of any segment, at least 2 rows should be aggregated to create a new segments that
        # approximate target_size. Therefore, the 1st row of a super segment has to be aggregated with the 2nd row.
        df.loc[(df["__new_seg__"] & ~df["__new_seg__"].shift(-1).fillna(True)), "__lim__"] = np.nan
        df["__lim__"] = df["__lim__"].bfill()

        # correction to reattach last isolated segments to the penultimate segments, if possible size wise
        df["__n_cut__"] = df["__n_cut__"].bfill()
        size_left = ((df["__n_cut__"] * target_size - (target_size // 1.5)) - df["__cumul__"].shift()).round()
        mask = (
                (df["__target__"] - target_size + size_left < 0.33 * target_size)  # check size after reattachment
                & (df["__cumul__"] > (df["__n_cut__"].floordiv(1) * df["__target__"]))  # check if last segments
                & (~((df["__%_min__"].diff() < 0) & (df["__%_min__"].diff(-1) < 0)))  # check if last segments
                & (~df["__new_seg__"])  # check if not super segment of single row
        )
        df.loc[mask, "__lim__"] = np.nan
        df["__lim__"] = df["__lim__"].ffill()

        #  2871305, 688214, 2347779, 1296188, 2535516, 2700249
        df = df.drop(
            ["__cumul__", "__new_seg__", "__n_cut__", "__target__", "__%_a__", "__%_b__", "__%_min__"],
            axis=1)
        return df["__lim__"]

def sort(df: pd.DataFrame, id_discrete: Iterable[Any], id_continuous: [Any, Any]) -> pd.DataFrame:
    return df.sort_values(by=[*id_discrete, *id_continuous])



def count_parallel_segment(df, id_discrete:Iterable[Any], id_continuous: [Any, Any]) -> pd.DataFrame:
    """This function aims at calculating the number of track for id_discret."""

    col = "parallel_count"
    df = df.__copy__()
    df_start = df.rename({id_continuous[0]: "___t___"}, axis=1).drop(columns=[id_continuous[1]])
    df_stop = df.rename({id_continuous[1]: "___t___"}, axis=1).drop(columns=[id_continuous[0]])
    df_start["count"] = 1
    df_stop["count"] = -1
    df_ret = pd.concat((df_start, df_stop), axis=0)
    df_ret.sort_values(by=[*id_discrete, '___t___'], inplace=True)
    df_ret[col] = df_ret.groupby(id_discrete)["count"].cumsum()
    df_inter = df_ret.groupby([*id_discrete, '___t___']).agg({col: "last"}).reset_index()

    change_nb_voie = df_inter[col].ne(df_inter[col].shift())
    start = df_inter.loc[change_nb_voie, "___t___"].to_numpy()
    stop = start[1:].tolist() + [df_inter["___t___"].iloc[-1]]

    df_inter = df_inter[change_nb_voie]
    df_inter[id_continuous[0]] =  start
    df_inter[id_continuous[1]] = stop

    df_inter = df_inter[start < stop]


    return df_inter[[*id_discrete, *id_continuous, col]]
