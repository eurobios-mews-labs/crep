crep.tools
==========

.. py:module:: crep.tools


Functions
---------

.. autoapisummary::

   crep.tools.build_admissible_data
   crep.tools.create_zones
   crep.tools.get_overlapping
   crep.tools.admissible_dataframe
   crep.tools.sample_non_admissible_data
   crep.tools.compute_discontinuity
   crep.tools.create_continuity
   crep.tools.create_continuity_modified
   crep.tools.cumul_length
   crep.tools.reorder_columns
   crep.tools.name_simplifier
   crep.tools.mark_new_segment
   crep.tools.cumul_segment_length
   crep.tools.concretize_aggregation
   crep.tools.n_cut_finder
   crep.tools.clusterize
   crep.tools.sort


Module Contents
---------------

.. py:function:: build_admissible_data(df: pandas.DataFrame, id_discrete: Iterable[Any], id_continuous: [Any, Any]) -> pandas.DataFrame

.. py:function:: create_zones(df: pandas.DataFrame, id_discrete: Iterable[Any], id_continuous: [Any, Any])

   
   Create overlapping zone identifiers in the DataFrame based on discrete and continuous ID columns.


   :Parameters:

       **df** : pd.DataFrame
           The input DataFrame containing the df.

       **id_discrete** : iter
           An iterable of column names that are considered discrete identifiers.

       **id_continuous** : iter
           An iterable of column names that are considered continuous identifiers.



   :Returns:

       pd.DataFrame
           The DataFrame with an additional '__zone__' column indicating the zone for each row.








   .. rubric:: Notes

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


   .. rubric:: Examples

   >>> df = {
   ...     'id': [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
   ...     't1': [932, 996, 2395, 2395, 3033, 3628, 4126, 4140, 4154, 316263, 316263, 316471, 316471],
   ...     't2': [2395, 2324, 3033, 3628, 3035, 4140, 4140, 5508, 5354, 316399, 316471, 317406, 317557],
   ...     'LONGUEUR': [1463, 1328, 638, 1233, 2, 512, 14, 1368, 1200, 136, 208, 935, 1086],
   ...     '__zone__': [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4]
   ... }
   >>> df = pd.DataFrame(df)
   >>> create_zones(df, ['id'], ['t1', 't2'])

   ..
       !! processed by numpydoc !!

.. py:function:: get_overlapping(df: pandas.DataFrame, id_discrete: Iterable[Any], id_continuous: [Any, Any]) -> pandas.Series

.. py:function:: admissible_dataframe(data: pandas.DataFrame, id_discrete: Iterable[Any], id_continuous: [Any, Any])

.. py:function:: sample_non_admissible_data(data: pandas.DataFrame, id_discrete: Iterable[Any], id_continuous: [Any, Any]) -> pandas.DataFrame

.. py:function:: compute_discontinuity(df, id_discrete: Iterable[Any], id_continuous: [Any, Any])

   
   Compute discontinuity in rail segment. The i-th element in return
   will be True if i-1 and i are discontinuous
















   ..
       !! processed by numpydoc !!

.. py:function:: create_continuity(df: pandas.DataFrame, id_discrete: Iterable[Any], id_continuous: [Any, Any], limit=None, sort=False) -> pandas.DataFrame

.. py:function:: create_continuity_modified(df: pandas.DataFrame, id_discrete: Iterable[Any], id_continuous: [Any, Any], limit=None, sort=False) -> pandas.DataFrame

.. py:function:: cumul_length(df: pandas.DataFrame, id_continuous: [Any, Any]) -> int

   
   Returns the sum of all segments sizes in the dataframe. 
















   ..
       !! processed by numpydoc !!

.. py:function:: reorder_columns(df: pandas.DataFrame, id_discrete: list[Any], id_continuous: [Any, Any])

.. py:function:: name_simplifier(names: list[str])

.. py:function:: mark_new_segment(df: pandas.DataFrame, id_discrete: list[Any], id_continuous: [Any, Any]) -> pandas.Series

   
   Creates a boolean pd.Series aligning with df indices. True: there is a change any of the id_discrete
   value between row n and row n-1 or there is a discontinuity (shown by id_continuous) between row n and row n-1
   Seems to be equivalent to crep.tools.compute_discontinuity


   :Parameters:

       **df** : pandas dataframe
           ..

       **id_discrete** : list
           discrete columns (object or categorical)

       **id_continuous** : list of 2 column names
           continuous columns that delimit the segments' start and end



   :Returns:

       df: boolean pandas series
           ..











   ..
       !! processed by numpydoc !!

.. py:function:: cumul_segment_length(df: pandas.DataFrame, id_discrete: list[Any], id_continuous: [Any, Any]) -> pandas.Series

   
   TODO : compute_cumulated_length
   Computes cumulative sum of segment length for each unique combination of id_discrete.


   :Parameters:

       **df** : pandas dataframe
           without duplicated rows or overlapping rows

       **id_discrete** : list
           discrete columns (object or categorical)

       **id_continuous** : list of 2 column names
           continuous columns that delimit the segments' start and end



   :Returns:

       df: pandas series with integers
           ..











   ..
       !! processed by numpydoc !!

.. py:function:: concretize_aggregation(df: pandas.DataFrame, id_discrete: list[Any], id_continuous: [Any, Any], dict_agg: dict[str, list[Any]] | None, add_group_by: Any | list[Any] = None, verbose: bool = False) -> pandas.DataFrame

   
   Groupby + aggregation operations


   :Parameters:

       **df** : pandas dataframe
           without duplicated rows or overlapping rows

       **id_discrete** : list
           discrete columns (object or categorical)

       **id_continuous** : list of 2 column names
           continuous columns that delimit the segments' start and end

       **dict_agg: dict, keys: agg operator, values: list of columns or None,**
           specify which aggregation operator to apply for which column. If None, default is mean for all columns.
           id_continuous, id_discrete and add_group_by columns don't need to be specified in the dictionary

       **add_group_by** : optional. column name or list of column names
           Additional columns to consider when grouping by

       **verbose: boolean**
           whether to print shape of df and if df is admissible at the end of the function.



   :Returns:

       **df** : pandas series with integers
           ..




   :Raises:

       Exception
           When the dataframe df passed in argument is not admissible i.e. it contains overlapping rows and or duplicates







   ..
       !! processed by numpydoc !!

.. py:function:: n_cut_finder(df: pandas.DataFrame, id_discrete: list[Any], id_continuous: [Any, Any], target_size: int, method: Literal['agg', 'split']) -> pandas.Series

   
   Finds in how many sub-segments the segment should be cut (method = "split") or find where to stop the aggregation of
   segments into a super segment (method = "agg"). The returned value of the function is the pd.Series of the column
    __n_cut__

   If method is "agg", the __n_cut__ contains non-NaN value everywhere but in the last row before a change of
   id_discrete value. The non-NaN values represent how many super-segments should result from the aggregation of the
   previous rows with NaN values.

   :Parameters:

       **df** : pandas dataframe
           without duplicated rows or overlapping rows

       **id_discrete** : list
           discrete columns (object or categorical)

       **id_continuous** : list of 2 column names
           continuous columns that delimit the segments' start and end

       **target_size: integer > 0**
           targeted segment size

       **method** : str, either "agg" or "split"
           Whether to find n_cut for aggregating (agg) or for splitting (split)



   :Returns:

       df: pandas series
           agg: series with floats and NaN. Floats are displayed in the rows that mark new segments.
           The remaining rows contain NaN. The float values indicates the number of possible target_sizes divisions in the
           segment (the sum of the previous NaN rows)
           split: series with integers >= 1. They indicate in how many segments the current row should be divided.




   :Raises:

       Exception
           When the dataframe df passed in argument is not admissible i.e. it contains overlapping rows and or duplicates







   ..
       !! processed by numpydoc !!

.. py:function:: clusterize(df: pandas.DataFrame, id_discrete: list[Any], id_continuous: [Any, Any], target_size: int) -> pandas.Series

   
   TODO: create_cluster_by_size
   Defines where to limit segment aggregation when uniformizing segment size to target size.


   :Parameters:

       **df** : pandas dataframe.
           The dataframe should be not have duplicated or overlapping rows.

       **id_discrete** : list
           discrete columns (object or categorical)

       **id_continuous** : list of 2 column names
           continuous columns that delimit the segments' start and end

       **target_size: integer > 0**
           targeted segment size



   :Returns:

       **df** : pandas series
           with common identifiers (integers) for the segments that should be grouped together.




   :Raises:

       Exception
           When the dataframe df passed in argument is not admissible i.e. it contains overlapping rows and or duplicates







   ..
       !! processed by numpydoc !!

.. py:function:: sort(df: pandas.DataFrame, id_discrete: list[Any], id_continuous: [Any, Any]) -> pandas.DataFrame

