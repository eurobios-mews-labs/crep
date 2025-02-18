import pandas as pd
from crep.table import DataFrameContinuous
from crep import tools, base


def test_constructor(get_examples):
    df_left, df_right = get_examples
    df = DataFrameContinuous(
        df_left,
        discrete_index=["id"],
        continuous_index=["t1", "t2"]
    )
    assert hasattr(df, "discrete_index")
    assert hasattr(df, "continuous_index")
    assert hasattr(df, "admissible")
    assert str(df) in DataFrameContinuous.instances

    # print(df**2)
    # assert hasattr(df**2, "discrete_index")


def test_getitem(get_examples_dataframe_continuous):
    df, _ = get_examples_dataframe_continuous
    new_df = df[["id", "t1", "t2"]]
    assert isinstance(new_df, DataFrameContinuous), type(new_df)


def test_concat_1(get_examples_dataframe_continuous):
    df_left, df_right = get_examples_dataframe_continuous
    res_internal_method = df_left.concat(other_dfs=df_right)
    res_external_method = pd.concat([df_left, df_right])
    assert (res_internal_method.shape == res_external_method.shape,
            (res_internal_method.shape,  res_external_method.shape))
    assert isinstance(res_internal_method, DataFrameContinuous), type(res_internal_method)


def test_concat_2(get_examples_dataframe_continuous):
    df_left, df_right = get_examples_dataframe_continuous
    res_internal_method = df_left.concat(other_dfs=df_right, axis=1)
    res_external_method = pd.concat([df_left, df_right], axis=1)
    assert (res_internal_method.shape == res_external_method.shape,
            (res_internal_method.shape,  res_external_method.shape))
    assert isinstance(res_internal_method, DataFrameContinuous), type(res_internal_method)


def test_reorder_columns(get_examples_dataframe_continuous):
    df, _ = get_examples_dataframe_continuous
    column_order = list(df.columns)
    df = df[["data1", "t2", "id", "t1"]]
    assert list(df.columns) != column_order
    df = df.reorder_columns()
    assert list(df.columns) == column_order
    assert isinstance(df, DataFrameContinuous), type(df)


def test_auto_sort(get_examples_dataframe_continuous):
    df_left, df_right = get_examples_dataframe_continuous
    df_left = df_left.concat(other_dfs=df_right)
    df_left = df_left.auto_sort()
    assert df_left["id"].iloc[2] == 1.0, df_left
    assert isinstance(df_left, DataFrameContinuous), type(df_left)


def test_filter_by_discrete_variable(get_advanced_examples_dataframe_continuous):
    df, _ = get_advanced_examples_dataframe_continuous
    df = df.filter_by_discrete_variables(dict_range={"id": [1], "id2": ["b"]})
    assert df.shape == (1, 5), df
    assert isinstance(df, DataFrameContinuous), type(df)


def test_filter_by_continuous_variable(get_advanced_examples_dataframe_continuous):
    _, df = get_advanced_examples_dataframe_continuous
    print(df.shape)
    df_new = df.filter_by_continuous_variables(dict_range={"data2": (0.1, 0.2)})
    assert df_new.shape == (7, 4), df_new.shape
    df_new = df.filter_by_continuous_variables(dict_range={"data2": (None, 0.2)})
    assert df_new.shape == (7, 4), df_new.shape
    df_new = df.filter_by_continuous_variables(dict_range={"data2": (0.2, None)})
    assert df_new.shape == (6, 4), df_new.shape
    assert isinstance(df, DataFrameContinuous), type(df)


def test_make_admissible(get_advanced_examples_dataframe_continuous):
    _, df = get_advanced_examples_dataframe_continuous
    assert not df.admissible
    df = df.make_admissible()
    assert df.admissible, df
    assert isinstance(df, DataFrameContinuous), type(df)


def test_create_continuity(get_advanced_examples_dataframe_continuous):
    _, df = get_advanced_examples_dataframe_continuous
    df_new = df.create_continuity(limit=50)
    assert len(df_new) > len(df), df_new
    df_new = df.create_continuity(limit=1)
    assert len(df_new) == len(df), df_new
    assert isinstance(df_new, DataFrameContinuous), type(df_new)


def test_crep_merge(get_examples_dataframe_continuous):
    df_left, df_right = get_examples_dataframe_continuous
    res_internal_method = df_left.crep_merge(
        data_right=df_right,
        how="outer"
    )
    res_external_method = base.merge(
        data_left=df_left,
        data_right=df_right,
        id_discrete=["id"],
        id_continuous=["t1", "t2"],
        how="outer"
    )
    assert str(res_internal_method) == str(res_external_method), (res_internal_method, res_external_method)
    assert isinstance(res_internal_method, DataFrameContinuous), type(res_internal_method)


def test_merge_event(get_examples_dataframe_continuous):
    df, _ = get_examples_dataframe_continuous
    event_data = pd.DataFrame({"id": [1, 2], "pk": [4, 105]})
    res_internal_method = df.merge_event(
        data_right=event_data,
        id_event="pk"
    )
    res_external_method = base.merge_event(
        data_left=df,
        data_right=event_data,
        id_discrete=["id"],
        id_continuous=["t1", "t2"],
        id_event="pk"
    )
    assert str(res_internal_method) == str(res_external_method), (res_internal_method, res_external_method)
    assert isinstance(res_internal_method, DataFrameContinuous), type(res_internal_method)


def test_aggregate_duplicates(get_advanced_examples_dataframe_continuous):
    _, df = get_advanced_examples_dataframe_continuous
    df = tools.build_admissible_data(
        df=df,
        id_discrete=["id"],
        id_continuous=["t1", "t2"]
    )
    df = DataFrameContinuous(df, discrete_index=["id"], continuous_index=["t1", "t2"])
    res_internal_method = df.aggregate_duplicates()
    res_external_method = base.aggregate_duplicates(
        df=df,
        id_discrete=["id"],
        id_continuous=["t1", "t2"]
    )
    assert str(res_internal_method) == str(res_external_method), (res_internal_method, res_external_method)
    assert isinstance(res_internal_method, DataFrameContinuous), type(res_internal_method)


def test_split_segment(get_examples_dataframe_continuous):
    df, _ = get_examples_dataframe_continuous
    res_internal_method = df.split_segment(
        target_size=10
    )
    res_external_method = base.split_segment(
        df=df,
        id_discrete=["id"],
        id_continuous=["t1", "t2"],
        target_size=10,
    )
    assert str(res_internal_method) == str(res_external_method), (res_internal_method, res_external_method)
    assert isinstance(res_internal_method, DataFrameContinuous), type(res_internal_method)


def test_homogenize(get_examples_dataframe_continuous):
    df, _ = get_examples_dataframe_continuous
    res_internal_method = df.homogenize(
        target_size=50
    )
    res_external_method = base.homogenize_within(
        df=df,
        id_discrete=["id"],
        id_continuous=["t1", "t2"],
        target_size=50,
    )
    print(res_internal_method)
    print(res_external_method)
    assert str(res_internal_method) == str(res_external_method), (res_internal_method, res_external_method)
    assert isinstance(res_internal_method, DataFrameContinuous), type(res_internal_method)

def test_aggregate_on_segmentation(get_examples_dataframe_continuous):
    df, _ = get_examples_dataframe_continuous
    df_data = df.homogenize(
        target_size=20
    )
    df_segmentation = df.split_segment(
        target_size=50,
    )
    df_segmentation = df_segmentation[["id", "t1", "t2"]]
    res_internal_method = df_data.aggregate_on_segmentation(
        df_segmentation=df_segmentation,
        dict_agg={"sum": ["data1"]}
    )
    res_external_method = base.aggregate_on_segmentation(
        df_segmentation=df_segmentation,
        df_data=df_data,
        id_discrete=["id"],
        id_continuous=["t1", "t2"],
        dict_agg={"sum": ["data1"]}
    )
    assert str(res_internal_method) == str(res_external_method), (res_internal_method, res_external_method)
    assert isinstance(res_internal_method, DataFrameContinuous), type(res_internal_method)