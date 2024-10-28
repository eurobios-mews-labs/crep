crep
====

.. py:module:: crep


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/crep/base/index
   /autoapi/crep/tools/index


Functions
---------

.. autoapisummary::

   crep.merge
   crep.aggregate_constant
   crep.unbalanced_merge
   crep.unbalanced_concat
   crep.homogenize_within
   crep.aggregate_duplicates
   crep.merge_event
   crep.compute_discontinuity


Package Contents
----------------

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

.. py:function:: homogenize_within(df: pandas.DataFrame, id_discrete: list[Any], id_continuous: [Any, Any], method: Literal['agg', 'split'] | list[Literal['agg', 'split']] | set[Literal['agg', 'split']] | None = None, target_size: None | int = None, dict_agg: dict[str, list[Any]] | None = None, strict_size: bool = False, verbose: bool = False) -> pandas.DataFrame

   
   Uniformizes segment size by splitting them into shorter segments close to target size. The uniformization aims
   to get a close a possible to target_size with +- 1.33 *  target_size as maximum error margin.


   :Parameters:

       **df** : pandas dataframe
           without duplicated rows or overlapping rows

       **id_discrete** : list
           discrete columns (object or categorical)

       **id_continuous** : list of 2 column names
           continuous columns that delimit the segments' start and end

       **method** : optional str, either "agg" or "split"
           Whether to homogenize segment length by splitting long segments ("split") or by aggregating short segments ("agg") or both.
           Default to None lets the function define the method.

       **target_size: optional, integer > 0 or None**
           targeted segment size. Default to None lets the function define the target size.

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











   ..
       !! processed by numpydoc !!

.. py:function:: aggregate_duplicates(df: pandas.DataFrame, id_discrete: list[Any], id_continuous: [Any, Any], dict_agg: dict[str, list[Any]] | None = None, verbose: bool = False)

   
   Removes duplicated rows by aggregating them.


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

.. py:function:: compute_discontinuity(df, id_discrete: Iterable[Any], id_continuous: [Any, Any])

   
   Compute discontinuity in rail segment. The i-th element in return
   will be True if i-1 and i are discontinuous
















   ..
       !! processed by numpydoc !!

