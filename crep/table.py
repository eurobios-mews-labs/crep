import pandas as pd
from typing import Any, Literal
import warnings

from crep import base, tools
from functools import wraps


def _ret(result, *args):
    if isinstance(result, DataFrameContinuous):
        return result
    elif isinstance(result, pd.DataFrame):
        if len(args) > 0 and isinstance(args[0], DataFrameContinuous):
            return DataFrameContinuous(result,
                                       discrete_index=args[0].discrete_index,
                                       continuous_index=args[0].continuous_index)
    return result


def modifier(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return _ret(result, *args)
    return wrapper


class DataFrameContinuous(pd.DataFrame):

    def __init__(
            self, *args,
            discrete_index,
            continuous_index: [Any, Any],
            **kwargs):
        super().__init__(*args, **kwargs)
        self.__discrete_index = discrete_index
        self.__continuous_index = continuous_index
        self.__checks()
        self.__overriding()

    def __checks(self):
        if len(self.__continuous_index) != 2:
            warnings.warn("the constructor must have 2 continuous index")
        for i in [*self.__continuous_index, *self.__discrete_index]:
            if i not in self.columns:
                warnings.warn(f"{i} must be in columns")

    def __make_func(self, attrib):
        """
        creates the functions that will override pd.DataFrame functions such as to return the current
        class instead of the parent class.
        """

        def func(*args, **kwargs):
            result = getattr(super(DataFrameContinuous, self), attrib)(*args, **kwargs)
            return _ret(result, self)

        return func


    def __overriding(self):
        """
        Goes through all attributes of the parent class and override the function, especially the return data type,
        when necessary. The running time of this function is 1ms, so 1s lost every 1000 new instances created.
        """

        for attrib in [func for func in dir(pd.DataFrame)]:
            if attrib not in ["__getitem__"]:
                if callable(getattr(pd.DataFrame, attrib)):
                    self.__dict__[attrib] = self.__make_func(attrib)


    def concat(self, other_dfs: pd.DataFrame | list[pd.DataFrame], **kwargs) -> 'DataFrameContinuous':
        """
        concat is an external function (not in the pd.DataFrame class, but called with pd.concat()
        This function builds the concat method such as to be able to call it as such: df.concat()
        """
        df = self
        if type(other_dfs) is not list:
            other_dfs = [other_dfs]
        new_df = pd.concat([df] + other_dfs, **kwargs)
        return DataFrameContinuous(new_df,
                                   discrete_index=self.__discrete_index,
                                   continuous_index=self.__continuous_index)

    def _return(self, df):
        return _ret(df, self)

    @modifier
    def reorder_columns(self):
        df = tools.reorder_columns(
            df=self,
            id_discrete=self.__discrete_index,
            id_continuous=self.__continuous_index
        )
        return df

    def auto_sort(self):
        df = self
        by_ = [col for col in [*df.discrete_index, *df.continuous_index] if col in self.columns]
        df = self.sort_values(by=by_).reset_index(drop=True)
        return df.reorder_columns()

    @modifier
    def filter_by_discrete_variables(self,
                                     dict_range: dict[str, tuple[Any | None, Any | None]]) -> 'DataFrameContinuous':
        """
        Filters a dataset by keeping only the specified values

        Parameters
        ----------
        dict_range: dict[str, tuple[Any, Any]]
        A dictionary with for keys the name of the variables and for values a tuple containing the minimum value
        to keep and the maximum value to keep
        """
        df = self
        for k, v in dict_range.items():
            mask = self[k].isin(v)
            df = df.loc[mask, :].reset_index(drop=True)
        return df

    @modifier
    def filter_by_continuous_variables(self,
                                       dict_range: dict[str, tuple[Any | None, Any | None]]) -> 'DataFrameContinuous':
        """
        Filter a dataset by keeping the values above, between or below continuous values

        Parameters
        ----------
        dict_range: dict[str, tuple[Any, Any]]
            A dictionary with for keys the name of the variables and for values a tuple containing the minimum value
            to keep and the maximum value to keep
        """
        df = self
        for k, v in dict_range.items():
            minimum, maximum = v
            if minimum is None and maximum is None:
                raise Exception("Error: to filter by date, the tuple with the range cannot be (None, None)")
            elif minimum is None:
                mask = df[k] <= maximum
            elif maximum is None:
                mask = df[k] >= minimum
            else:
                mask = (df[k] >= minimum) & (df[k] <= maximum)
            df = df.loc[mask, :].reset_index(drop=True)
        return df

    def make_admissible(self, verbose=False):
        df = self
        if not self.admissible:
            df = tools.build_admissible_data(
                df=df,
                id_discrete=self.__discrete_index,
                id_continuous=self.__continuous_index
            )
        is_admissible = tools.admissible_dataframe(
            data=df,
            id_discrete=self.__discrete_index,
            id_continuous=self.__continuous_index
        )
        if not is_admissible:
            warnings.warn("Function aggregate_duplicates used with 'mean' as default aggregation operator "
                          "in order to make the dataframe admissible.")
            df = base.aggregate_duplicates(
                df=df,
                id_discrete=self.__discrete_index,
                id_continuous=self.__continuous_index
            )
        if verbose:
            print("post make_admissible. Admissible:", df.admissible)
            print(df.shape)
        df = self._return(df)
        df = df.auto_sort()
        return df

    def create_continuity(self, limit=None, sort=False) -> 'DataFrameContinuous':
        df = tools.create_continuity(
            df=self,
            id_discrete=self.__discrete_index,
            id_continuous=self.__continuous_index,
            limit=limit,
            sort=sort
        )
        return self._return(df)

    def crep_merge(
            self,
            data_right: pd.DataFrame,
            how: str,
            remove_duplicates: bool = False,
            verbose: bool = False
    ) -> 'DataFrameContinuous':
        df = base.merge(
            data_left=self,
            data_right=data_right,
            id_discrete=self.__discrete_index,
            id_continuous=self.__continuous_index,
            how=how,
            remove_duplicates=remove_duplicates,
            verbose=verbose
        )
        return self._return(df)

    def merge_event(self, data_right: pd.DataFrame, id_event) -> 'DataFrameContinuous':
        df = base.merge_event(
            data_left=self,
            data_right=data_right,
            id_discrete=self.__discrete_index,
            id_continuous=self.__continuous_index,
            id_event=id_event
        )
        return self._return(df)

    def aggregate_duplicates(
            self,
            dict_agg: None | dict[str, list[Any]] = None,
            verbose: bool = False
    ) -> 'DataFrameContinuous':
        df = base.aggregate_duplicates(
            df=self,
            id_discrete=self.__discrete_index,
            id_continuous=self.__continuous_index,
            dict_agg=dict_agg,
            verbose=verbose
        )
        return self._return(df)

    def aggregate_continuous_data(
            self,
            target_size: int,
            dict_agg: None | dict[str, list[Any]] = None,
            verbose: bool = False
    ) -> 'DataFrameContinuous':
        df = base.aggregate_continuous_data(
            df=self,
            id_discrete=self.__discrete_index,
            id_continuous=self.__continuous_index,
            target_size=target_size,
            dict_agg=dict_agg,
            verbose=verbose
        )
        return self._return(df)

    def split_segment(
            self,
            target_size: int,
            columns_sum_aggregation: list[str] = None,
            verbose: bool = False
    ) -> 'DataFrameContinuous':
        df = base.split_segment(
            df=self,
            id_discrete=self.__discrete_index,
            id_continuous=self.__continuous_index,
            target_size=target_size,
            columns_sum_aggregation=columns_sum_aggregation,
            verbose=verbose
        )
        return self._return(df)

    def homogenize(
            self,
            target_size: int,
            method: Literal["agg", "split"] | list[Literal["agg", "split"]] | set[
                Literal["agg", "split"]] | None = None,
            dict_agg: dict[str, list[Any]] | None = None,
            strict_size: bool = False,
            verbose: bool = False
    ) -> 'DataFrameContinuous':
        df = base.homogenize_within(
            df=self,
            id_discrete=self.__discrete_index,
            id_continuous=self.__continuous_index,
            target_size=target_size,
            method=method,
            dict_agg=dict_agg,
            strict_size=strict_size,
            verbose=verbose
        )
        return self._return(df)

    def aggregate_on_segmentation(
            self,
            df_segmentation: pd.DataFrame,
            dict_agg: dict[str, list[str]] | None = None
    ) -> 'DataFrameContinuous':
        if len(df_segmentation.columns) > len(self.__discrete_index) + len(self.__continuous_index):
            warnings.warn("df_segmentation contains more columns than necessary. "
                          "Other columns than discrete or continuous indices are dropped.")
            df_segmentation = df_segmentation[[*self.__discrete_index, *self.__continuous_index]]
        df = base.aggregate_on_segmentation(
            df_segmentation=df_segmentation,
            df_data=self,
            id_discrete=self.__discrete_index,
            id_continuous=self.__continuous_index,
            dict_agg=dict_agg,
        )
        return self._return(df)

    @property
    def discrete_index(self):
        return self.__discrete_index

    @property
    def continuous_index(self):
        return self.__continuous_index

    @property
    def admissible(self):
        return tools.admissible_dataframe(
            self,
            self.__discrete_index,
            self.__continuous_index
        )
