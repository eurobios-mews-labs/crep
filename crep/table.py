# from encodings.punycode import selective_find

import pandas as pd
from typing import Any, Literal
import warnings

from crep import base, tools


class DataFrameContinuous(pd.DataFrame):
    instances = []

    def __init__(
            self, *args,
            discrete_index,
            continuous_index: [Any, Any],
            **kwargs):
        super().__init__(*args, **kwargs)
        self._discrete_index = discrete_index
        self._continuous_index = continuous_index
        self._checks()
        self._overriding()

    def _checks(self):
        if len(self._continuous_index) != 2:
            warnings.warn("the constructor must have 2 continuous index")
        for i in [*self._continuous_index, *self._discrete_index]:
            if i not in self.columns:
                warnings.warn(f"{i} must be in columns")

    def _make_func(self, attrib):
        """
        creates the functions that will override pd.DataFrame functions such as to return the current
        class instead of the parent class.
        """
        def func(*args, **kwargs):
            result = getattr(super(DataFrameContinuous, self), attrib)(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                return DataFrameContinuous(result,
                                           discrete_index=self._discrete_index,
                                           continuous_index=self._continuous_index)
            return result
        return func

    def _overriding(self):
        """
        Goes through all attributes of the parent class and override the function, especially the return data type,
        when necessary. The running time of this function is 1ms, so 1s lost every 1000 new instances created.
        """
        if str(self) not in DataFrameContinuous.instances:
            for attrib in [func for func in dir(pd.DataFrame)]:
                if attrib not in ["__getitem__"]:
                    if callable(getattr(pd.DataFrame, attrib)):
                        self.__dict__[attrib] = self._make_func(attrib)
            DataFrameContinuous.instances.append(str(self))

    def __getitem__(self, key):
        """
        Also needs to be modified since df[key] can be used to return a subsample of the df
        """
        result = getattr(super(DataFrameContinuous, self), "__getitem__")(key)
        if isinstance(result, pd.DataFrame):
            return DataFrameContinuous(result,
                                       discrete_index=self._discrete_index,
                                       continuous_index=self._continuous_index)
        return result

    def _return(self, df):
        if isinstance(df, DataFrameContinuous):
            return df
        else:
            return DataFrameContinuous(df,
                                       discrete_index=self._discrete_index,
                                       continuous_index=self._continuous_index)

    def concat(self, other_dfs: pd.DataFrame | list[pd.DataFrame], **kwargs) -> 'DataFrameContinuous':
        """
        concat is an external function (not in the pd.DataFrame class, but called with pd.concat()
        This function builds the concat method such as to be able to call it as such: df.concat()
        """
        df = self
        if type(other_dfs) is not list:
            other_dfs = [other_dfs]
        new_df = pd.concat([df]+other_dfs, **kwargs)
        return self._return(new_df)

    def reorder_columns(self):
        df = self
        df = tools.reorder_columns(
            df=df,
            id_discrete=self._discrete_index,
            id_continuous=self._continuous_index
        )
        return self._return(df)

    def auto_sort(self):
        df = self
        by_ = [col for col in [*df.discrete_index, *df.continuous_index] if col in self.columns]
        df = self.sort_values(by=by_).reset_index(drop=True)
        df = self._return(df)
        df = df.reorder_columns()
        return self._return(df)

    def filter_by_discrete_variables(
            self,
            dict_range: dict[str, tuple[Any | None, Any | None]],
            keep_nan: bool = False
    ) -> 'DataFrameContinuous':
        """
        Filters a dataset by keeping only the specified values

        Parameters
        ----------
        dict_range: dict[str, tuple[Any, Any]]
            A dictionary with for keys the name of the variables and for values a tuple containing the minimum value
            to keep and the maximum value to keep
        keep_nan: optional, default to False
            if True, rows with nan are kept
        """
        df = self
        for k, v in dict_range.items():
            mask = df[k].isin(v)
            if keep_nan:
                mask = mask | df[k].isna()
            df = df.loc[mask, :].reset_index(drop=True)
        return self._return(df)

    def filter_by_continuous_variables(
            self,
            dict_range: dict[str, tuple[Any | None, Any | None]],
            keep_nan: bool = True,
    ) -> 'DataFrameContinuous':
        """
        Filter a dataset by keeping the values above, between or below continuous values

        Parameters
        ----------
        dict_range: dict[str, tuple[Any, Any]]
            A dictionary with for keys the name of the variables and for values a tuple containing the minimum value
            to keep and the maximum value to keep
        keep_nan: optional, default to True
            if True, rows with nan are kept
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
            if keep_nan:
                mask = mask | df[k].isna()
            df = df.loc[mask, :].reset_index(drop=True)
        return self._return(df)

    def make_admissible(self, verbose=False):
        df = self
        if not self.admissible:
            df = tools.build_admissible_data(
                df=df,
                id_discrete=self._discrete_index,
                id_continuous=self._continuous_index
            )
        is_admissible = tools.admissible_dataframe(
                data=df,
                id_discrete=self._discrete_index,
                id_continuous=self._continuous_index
        )
        if not is_admissible:
            warnings.warn("Function aggregate_duplicates used with 'mean' as default aggregation operator "
                          "in order to make the dataframe admissible.")
            df = base.aggregate_duplicates(
                df=df,
                id_discrete=self._discrete_index,
                id_continuous=self._continuous_index
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
            id_discrete=self._discrete_index,
            id_continuous=self._continuous_index,
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
            id_discrete=self._discrete_index,
            id_continuous=self._continuous_index,
            how=how,
            remove_duplicates=remove_duplicates,
            verbose=verbose
        )
        return self._return(df)

    def merge_event(self, data_right: pd.DataFrame, id_event) -> 'DataFrameContinuous':
        df = base.merge_event(
            data_left=self,
            data_right=data_right,
            id_discrete=self._discrete_index,
            id_continuous=self._continuous_index,
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
            id_discrete=self._discrete_index,
            id_continuous=self._continuous_index,
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
                id_discrete=self._discrete_index,
                id_continuous=self._continuous_index,
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
            id_discrete=self._discrete_index,
            id_continuous=self._continuous_index,
            target_size=target_size,
            columns_sum_aggregation=columns_sum_aggregation,
            verbose=verbose
        )
        return self._return(df)

    def homogenize(
            self,
            target_size: int,
            method:  Literal["agg", "split"] | list[Literal["agg", "split"]] | set[Literal["agg", "split"]] | None = None,
            dict_agg: dict[str, list[Any]] | None = None,
            strict_size: bool = False,
            verbose: bool = False
    ) -> 'DataFrameContinuous':
        df = base.homogenize_within(
            df=self,
            id_discrete=self._discrete_index,
            id_continuous=self._continuous_index,
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
        if len(df_segmentation.columns) > len(self._discrete_index) + len(self._continuous_index):
            warnings.warn("df_segmentation contains more columns than necessary. "
                          "Other columns than discrete or continuous indices are dropped.")
            df_segmentation = df_segmentation[[*self._discrete_index, *self._continuous_index]]
        df = base.aggregate_on_segmentation(
            df_segmentation=df_segmentation,
            df_data=self,
            id_discrete=self._discrete_index,
            id_continuous=self._continuous_index,
            dict_agg=dict_agg,
        )
        return self._return(df)

    @property
    def discrete_index(self):
        return self._discrete_index

    @property
    def continuous_index(self):
        return self._continuous_index

    @property
    def admissible(self):
        return tools.admissible_dataframe(
            self,
            self._discrete_index,
            self._continuous_index
        )