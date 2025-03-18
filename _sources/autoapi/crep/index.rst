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
   crep.aggregate_duplicates
   crep.merge_event
   crep.aggregate_on_segmentation
   crep.unbalanced_merge
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

.. py:function:: aggregate_constant(df: pandas.DataFrame, id_discrete: Iterable[Any], id_continuous: [Any, Any])

   



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

.. py:function:: aggregate_duplicates(df: pandas.DataFrame, id_discrete: Iterable[Any], id_continuous: [Any, Any], dict_agg: Optional[Dict[str, Iterable[Any]]] = None, verbose: bool = False)

   
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

.. py:function:: merge_event(data_left: pandas.DataFrame, data_right: pandas.DataFrame, id_discrete: Iterable[Any], id_continuous: [Any, Any], id_event)

   
   Assigns the details of events occurring at a specific points, in data_right, to the corresponding segment
   in data_left.


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

.. py:function:: aggregate_on_segmentation(df_segmentation: pandas.DataFrame, df_data: pandas.DataFrame, id_discrete: Iterable[str], id_continuous: Iterable[str], dict_agg: Optional[Dict[str, Iterable[str]]] = None)

   
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

.. py:function:: unbalanced_merge(data_admissible: pandas.DataFrame, data_not_admissible: pandas.DataFrame, id_discrete: Iterable, id_continuous: [Any, Any], how) -> pandas.DataFrame

   
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

       **how: str**
           how to make the merge, possible options are
           
           - 'left'
           - 'right'
           - 'inner'
           - 'outer'



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

.. py:function:: compute_discontinuity(df, id_discrete: Iterable[Any], id_continuous: [Any, Any])

   
   Compute discontinuity in rail segment. The i-th element in return
   will be True if i-1 and i are discontinuous
















   ..
       !! processed by numpydoc !!

