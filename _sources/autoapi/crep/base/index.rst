crep.base
=========

.. py:module:: crep.base


Functions
---------

.. autoapisummary::

   crep.base.merge
   crep.base.unbalanced_merge
   crep.base.unbalanced_concat
   crep.base.aggregate_constant
   crep.base.__merge_index
   crep.base.merge_event
   crep.base.create_regular_segmentation
   crep.base.__merge
   crep.base.__fix_discrete_index
   crep.base.suppress_duplicates
   crep.base._increasing_continuous_index
   crep.base.__refactor_data
   crep.base.__check_args_merge
   crep.base.__table_jumps
   crep.base.aggregate_duplicates
   crep.base.aggregate_continuous_data
   crep.base.split_segment
   crep.base.homogenize_within
   crep.base.homogenize_between
   crep.base.segmentation_irregular
   crep.base.segmentation_regular
   crep.base.aggregate_on_segmentation


Module Contents
---------------

.. py:function:: merge(data_left: pandas.DataFrame, data_right: pandas.DataFrame, id_continuous: [Any, Any], id_discrete: iter, how: str, remove_duplicates: bool = False, verbose=False) -> pandas.DataFrame

   
   This function aims at creating merge data frame


   :Parameters:

       **data_left**
           data frame with continuous representation

       **data_right**
           data frame with continuous representation

       **id_continuous**
           iterable of length two that delimits the edges of the segment

       **id_discrete: iterable**
           iterable that lists all the columns on which to perform a classic merge

       **how: str**
           how to make the merge, possible options are
           
           - 'left'
           - 'right'
           - 'inner'
           - 'outer'

       **remove_duplicates**
           whether to remove duplicates

       **verbose**
           ..














   ..
       !! processed by numpydoc !!

.. py:function:: unbalanced_merge(data_admissible: pandas.DataFrame, data_not_admissible: pandas.DataFrame, id_discrete: iter, id_continuous: [Any, Any]) -> pandas.DataFrame

   
   Merge admissible and non-admissible dataframes based on discrete and continuous identifiers.


   :Parameters:

       **data_admissible** : pd.DataFrame
           DataFrame containing admissible data.

       **data_not_admissible** : pd.DataFrame
           DataFrame containing non-admissible data.

       **id_discrete** : list
           List of column names representing discrete identifiers.

       **id_continuous** : list
           List of column names representing continuous identifiers.



   :Returns:

       pd.DataFrame
           A DataFrame resulting from the unbalanced merge of admissible and non-admissible data.








   .. rubric:: Notes

   The function performs the following steps:
   1. Combines and sorts the admissible and non-admissible data based on the identifiers.
   2. Resolves overlaps and conflicts between the admissible and non-admissible data.
   3. Merges and returns the final DataFrame.



   ..
       !! processed by numpydoc !!

.. py:function:: unbalanced_concat(df1: pandas.DataFrame, df2: pandas.DataFrame, id_discrete: list[Any], id_continuous: [Any, Any], ignore_homogenize: bool = False, verbose: bool = False) -> pandas.DataFrame

   
   Concatenates the rows from two dataframes, and adjusts the lengths of the segments so that for each segment in the
   first dataframe there is a segment in the second dataframes with the same id_continuous characteristics, and
   vice versa. This function can handle duplicated rows in each other of the df, but not non-duplicated overlap.


   :Parameters:

       **df1** : pandas dataframe
           ..

       **df2** : pandas dataframe
           ..

       **id_discrete** : list
           discrete columns (object or categorical)

       **id_continuous** : list of 2 column names
           continuous columns that delimit the segments' start and end

       **ignore_homogenize** : optional. boolean
           if True, ignore the homogenization function

       **verbose: optional. boolean**
           whether to print shape of df and if df is admissible at the end of the function.



   :Returns:

       df:  pandas dataframe
           ..











   ..
       !! processed by numpydoc !!

.. py:function:: aggregate_constant(df: pandas.DataFrame, id_discrete: iter, id_continuous: iter)

   



   :Parameters:

       **df**
           ..

       **id_discrete**
           ..

       **id_continuous**
           ..



   :Returns:

       
           ..











   ..
       !! processed by numpydoc !!

.. py:function:: __merge_index(data_left, data_right, id_discrete, id_continuous, names=('left', 'right'))

.. py:function:: merge_event(data_left: pandas.DataFrame, data_right: pandas.DataFrame, id_discrete: iter, id_continuous: [Any, Any], id_event)

   
   Merges two dataframes on both discrete and continuous indices, with forward-filling of missing data.

   This function merges two Pandas DataFrames (`data_left` and `data_right`) based on discrete and continuous keys.
   It assigns the event data from data_right to the correct segment in data_left, if the event is not "out-of-bound"
   relative to the segments in data_left. The result is a dataframe with a new row for each event. Rows with NaN
   event data are kept to represent the segment state prior to the occurrence of any event (as such the returned
   dataframe contains duplicates based on subsets of columns id_discrete and id_continuous).

   :Parameters:

       **data_left** : pd.DataFrame
           The left dataframe to be merged.

       **data_right** : pd.DataFrame
           The right dataframe to be merged.

       **id_discrete** : iterable
           The list of column names representing discrete identifiers for sorting and merging
           (e.g., categorical variables)

       **id_continuous** : list of two elements (Any, Any)
           A list with two elements representing the continuous index (e.g., time or numerical variables).
           The first element is the column name of the continuous identifier used for sorting.

       **id_event:**
           the name of the column containing the exact localisation of the event



   :Returns:

       pd.DataFrame
           A merged dataframe that combines `data_left` and `data_right`.











   ..
       !! processed by numpydoc !!

.. py:function:: create_regular_segmentation(data: pandas.DataFrame, length, id_discrete: iter, id_continuous: [Any, Any]) -> pandas.DataFrame

   
   Creates a regular segmentation of a DataFrame based on specified discrete and continuous columns,
   segmenting continuous values into equal-length intervals.


   :Parameters:

       **data** : pd.DataFrame
           Input DataFrame containing the data to be segmented.

       **length** : int
           Length of each segment in the continuous interval. If set to 0, returns the original DataFrame.

       **id_discrete** : iterable
           Column names to be used as discrete identifiers for grouping data.

       **id_continuous** : list of Any, Any
           A list containing two continuous column names, where the first item represents the start value
           and the second item represents the end value for the segmentation.



   :Returns:

       pd.DataFrame
           A new DataFrame with regular segments created based on the specified interval length and grouping columns.
           The resulting DataFrame includes columns for the discrete identifiers, the continuous start and end values
           for each segment.




   :Raises:

       ValueError
           If `length` is not a positive integer.




   .. rubric:: Notes

   The function calculates the number of segments for each discrete group by dividing the interval between the
   maximum and minimum continuous values by the given segment length. Segments are then evenly spaced within
   this interval for each discrete group.



   ..
       !! processed by numpydoc !!

.. py:function:: __merge(df_left: pandas.DataFrame, df_right: pandas.DataFrame, id_discrete: iter, id_continuous, names=('left', 'right'))

.. py:function:: __fix_discrete_index(data_left: pandas.DataFrame, data_right: pandas.DataFrame, id_discrete_left: iter, id_discrete_right: iter)

.. py:function:: suppress_duplicates(df, id_discrete, continuous_index)

.. py:function:: _increasing_continuous_index(df: pandas.DataFrame, id_continuous: [Any, Any])

.. py:function:: __refactor_data(data_left, data_right, id_continuous, id_discrete, names=('left', 'right'))

.. py:function:: __check_args_merge(data_left, data_right, id_continuous, id_discrete, how)

.. py:function:: __table_jumps(data, id1, id2, id_discrete)

.. py:function:: aggregate_duplicates(df: pandas.DataFrame, id_discrete: list[Any], id_continuous: [Any, Any], dict_agg: dict[str, list[Any]] | None = None, verbose: bool = False)

   
   Removes duplicated rows by aggregating them.
   TODO : assess


   :Parameters:

       **df** : pandas dataframe
           ..

       **id_discrete** : list
           discrete columns (object or categorical)

       **id_continuous** : list of 2 column names
           continuous columns that delimit the segments' start and end

       **dict_agg: dict, keys: agg operator, values: list of columns or None**
           specify which aggregation operator to apply for which column. If None, default is mean for all columns.
           id_continuous and id_discrete columns don't need to be specified in the dictionary

       **verbose: boolean**
           whether to print shape of df and if df is admissible at the end of the function.



   :Returns:

       df: pandas dataframe
           without duplicated rows




   :Raises:

       Exception
           When the dataframe df passed in argument does not contain any duplicated row







   ..
       !! processed by numpydoc !!

.. py:function:: aggregate_continuous_data(df: pandas.DataFrame, id_discrete: list[Any], id_continuous: [Any, Any], target_size: int, dict_agg: None | dict[str, list[Any]] = None, verbose: bool = False) -> pandas.DataFrame

   
   Aggregate segments to uniformize the size of smaller segments.


   :Parameters:

       **df** : pandas dataframe
           without duplicated rows or overlapping rows

       **id_discrete** : list
           discrete columns (object or categorical)

       **id_continuous** : list of 2 column names
           continuous columns that delimit the segments' start and end

       **target_size: integer > 0**
           targeted segment size

       **dict_agg: optional. dict, keys: agg operator, values: list of columns or None,**
           specify which aggregation operator to apply for which column. If None, default is mean for all columns.
           id_continuous, id_discrete and add_group_by columns don't need to be specified in the dictionary

       **verbose: optional. boolean**
           whether to print shape of df and if df is admissible at the end of the function.



   :Returns:

       df: pandas dataframe
           ..




   :Raises:

       Exception
           When the dataframe df passed in argument is not admissible i.e. it contains overlapping rows and or duplicates







   ..
       !! processed by numpydoc !!

.. py:function:: split_segment(df: pandas.DataFrame, id_discrete: list[Any], id_continuous: [Any, Any], target_size: int, col_sum_agg: list[str] = None, verbose: bool = False) -> pandas.DataFrame

   
   Uniformizes segment size by splitting them into shorter segments close to target size.


   :Parameters:

       **df** : pandas dataframe
           without duplicated rows or overlapping rows

       **id_discrete** : list
           discrete columns (object or categorical)

       **id_continuous** : list of 2 column names
           continuous columns that delimit the segments' start and end

       **target_size: integer > 0**
           targeted segment size

       **col_sum_agg: list[str], optional**
           Default to empty list. Some columns may have to be summed over several segments when creating super segments.
           If so, splitting a row and assigning to each new row the same value as in the original non-split row may
           result in inflated sums later on. To counter that, the columns that should later be summed are specified in
           this list. The values are transformed into ratios relative to the segment size, then the row is split, and
           then an inverse transformation is done to reassign a non-ratio value.

       **verbose: optional. boolean**
           whether to print shape of df and if df is admissible at the end of the function.



   :Returns:

       df: pandas dataframe
           ..











   ..
       !! processed by numpydoc !!

.. py:function:: homogenize_within(df: pandas.DataFrame, id_discrete: list[Any], id_continuous: [Any, Any], target_size: float | int | None = None, method: Literal['agg', 'split'] | list[Literal['agg', 'split']] | set[Literal['agg', 'split']] | None = None, dict_agg: dict[str, list[Any]] | None = None, strict_size: bool = False, verbose: bool = False) -> pandas.DataFrame

   
   Uniformizes segment size by splitting them into shorter segments close to target size. The uniformization aims
   to get a close a possible to target_size with +- 1.33 * target_size as maximum error margin.


   :Parameters:

       **df** : pandas dataframe
           without duplicated rows or overlapping rows

       **id_discrete** : list
           discrete columns (object or categorical)

       **id_continuous** : list of 2 column names
           continuous columns that delimit the segments' start and end

       **target_size: optional, integer > 0 or None**
           targeted segment size. If None, the median is selected.

       **method** : optional str, either "agg" or "split"
           Whether to homogenize segment length by splitting long segments ("split") or by aggregating short segments ("agg") or both.
           Default to None lets the function define the method.

       **strict_size: whether to strictly respect target_size specified in argument, if any specified.**
           The function can change the target size if the value is not congruent with the method

       **dict_agg: optional. dict, keys: agg operator, values: list of columns or None,**
           specify which aggregation operator to apply for which column. If None, default is mean for all columns.
           id_continuous, id_discrete and add_group_by columns don't need to be specified in the dictionary

       **verbose: optional. boolean**
           whether to print shape of df and if df is admissible at the end of the function.



   :Returns:

       df: pandas dataframe
           ..




   :Raises:

       Exception:
           If method is not defined and if the function failed to select automatically a method.







   ..
       !! processed by numpydoc !!

.. py:function:: homogenize_between(df1: pandas.DataFrame, df2: pandas.DataFrame, id_discrete: list[Any], id_continuous: list[Any], dict_agg_df1: dict[str, list[str]] | None = None, dict_agg_df2: dict[str, list[str]] | None = None, keep_df1: bool = False, verbose: bool = False) -> tuple[pandas.DataFrame, pandas.DataFrame]

   
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

   :Parameters:

       **df1** : pandas dataframe
           ..

       **df2** : pandas dataframe
           ..

       **id_discrete** : list
           discrete columns (object or categorical)

       **id_continuous** : list of 2 column names
           continuous columns that delimit the segments' start and end

       **dict_agg_df1: optional, dict[str, list[str]] | None**
           dictionary with settings about how to handle the columns in df1 that are neither id_discrete nor id_continuous

       **dict_agg_df2: optional, dict[str, list[str]] | None**
           dictionary with settings about how to handle the columns in df2 that are neither id_discrete nor id_continuous

       **keep_df1: optional, bool**
           default to False. If True, the segmentation in df1 does not change. Only df2 adapts to df1.

       **verbose: optional. boolean**
           whether to print shape of df and if df is admissible at the end of the function.



   :Returns:

       df: pandas dataframe
           ..











   ..
       !! processed by numpydoc !!

.. py:function:: segmentation_irregular(df: pandas.DataFrame, id_discrete: list[Any], id_continuous: [Any, Any], length_target, length_minimal) -> pandas.DataFrame

   



   :Parameters:

       **df: pd.DataFrame**
           ..

       **id_discrete: list[str]**
           list of name of columns of categorical type

       **id_continuous: list[str, str]**
           list of name of 2 columns of numerical type, indicating the start and the end of the segment

       **length_target**
           length to obtain at the end of the segmentation

       **length_minimal**
           When there are gaps in the dataframe, define the length beyond which this could be considered as a
           deliberate break in the segmentation and not as missing data. Under this threshold, a new row will
           be created to ensure the continuity between successive segments in the dataframe.



   :Returns:

       pd.DataFrame
           New dataframe containing only the columns id_discrete and id_continuous, with the length of the segments
           adjusted to be as close as possible to length_target.











   ..
       !! processed by numpydoc !!

.. py:function:: segmentation_regular(df: pandas.DataFrame, id_discrete: list[Any], id_continuous: [Any, Any], length_target, length_gap_filling) -> pandas.DataFrame

.. py:function:: aggregate_on_segmentation(df_segmentation: pandas.DataFrame, df_data: pandas.DataFrame, id_discrete: list[str], id_continuous: list[str], dict_agg: dict[str, list[str]] | None = None)

   
   adds data to segmentation


   :Parameters:

       **df_segmentation: pd.DataFrame**
           the dataframe containing the segmentation. Should contain only columns id_discrete and id_continuous

       **df_data: pd.DataFrame**
           the dataframe containing the features to fit to the segmentation. Should contain the columns
           id_discrete and id_continuous as well as other columns for the features of interest.

       **id_discrete**
           ..

       **id_continuous**
           ..

       **dict_agg:**
           ..



   :Returns:

       pd.DataFrame:
           a dataframe with the feature data fitted to the new segmentation.











   ..
       !! processed by numpydoc !!

