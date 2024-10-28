
from crep.table import DataFrameContinuous


def test_constructor(get_examples):
    df_left, df_right = get_examples
    df = DataFrameContinuous(
        df_left,
        discrete_index=["id"],
        continuous_index=["t1", "t2"])
    assert hasattr(df, "discrete_index")
    assert hasattr(df, "continuous_index")
    assert hasattr(df, "admissible")
