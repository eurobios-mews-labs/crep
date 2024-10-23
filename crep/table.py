from pandas import DataFrame
from crep import tools


from typing import Any


class DataFrameContinuous(DataFrame):

    def __init__(
            self, *args,
            discrete_index,
            continuous_index: [Any, Any],
            **kwargs):
        super().__init__(*args, **kwargs)
        self.__discrete_index = discrete_index
        self.__continuous_index = continuous_index
        self.__checks()

    def __checks(self):
        if len(self.__continuous_index) != 2:
            raise ValueError("the constructor must have "
                             "2 continuous index")
        for i in [*self.__continuous_index, *self.__discrete_index]:
            if i not in self.columns:
                raise ValueError(f"{i} must be in columns")

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
            self.__continuous_index)
