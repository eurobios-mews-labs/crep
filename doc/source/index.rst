.. Continuous data REPresentation documentation master file, created by
   sphinx-quickstart on Mon Oct 28 18:05:08 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Tabular data processing for Continuous REPresentation
=====================================================

.. image:: https://github.com/eurobios-scb/crep/actions/workflows/pytest.yml/badge.svg?event=push
   :target: https://docs.pytest.org
.. image:: https://raw.githubusercontent.com/eurobios-mews-labs/crep/coverage-badge/coverage.svg?raw=true
   :alt: code coverage

To install, run:

.. code-block:: python

    pip install crep

This simple module provides functions for handling tabular data with a continuous axis. In certain cases, this index can represent time, but this tool was initially developed to address railway descriptions.

The tool is designed to represent linear structures (cables, rails, beams, pipes) with piece-wise constant characteristics, even when the segment lengths are highly variable.

Basic usage
-----------

**Merge function**: Merges two DataFrames.

.. code-block:: python

    import pandas as pd
    from crep import merge

    df_left = pd.DataFrame(
        dict(id=[2, 2, 2],
             t1=[0, 100, 120],
             t2=[100, 120, 130],
             data1=[0.2, 0.1, 0.5])
    )
    df_right = pd.DataFrame(
        dict(id=[2, 2, 2],
             t1=[0, 80, 100],
             t2=[70, 100, 140],
             data2=[0.1, 0.3, 0.2])
    )

    ret = merge(data_left=df_left,
                data_right=df_right,
                id_continuous=["t1", "t2"],
                id_discrete=["id"],
                how="outer")

The resulting output is:

.. image:: ../../examples/basic_example.png
   :alt: basic_example
   :width: 350

Tools
-----

To check if your data is admissible for the merge function, you can use the `tools` module.

.. code-block:: python

    import pandas as pd
    from crep import tools

    df_admissible = pd.DataFrame(
        dict(id=[2, 2, 2],
             t1=[0, 100, 120],
             t2=[100, 120, 130],
             data1=[0.2, 0.1, 0.5])
    )
    df_not_admissible = pd.DataFrame(
        dict(id=[2, 2, 2],
             t1=[0, 90, 120],
             t2=[100, 120, 130],
             data1=[0.2, 0.1, 0.5])
    )
    # The second table is not admissible because two values are possible for t in [90,100].

    assert tools.admissible_dataframe(
        df_admissible, id_continuous=["t1", "t2"],
        id_discrete=["id"])
    assert not tools.admissible_dataframe(
        df_not_admissible, id_continuous=["t1", "t2"],
        id_discrete=["id"])
    print(tools.sample_non_admissible_data(
        df_not_admissible, id_continuous=["t1", "t2"],
        id_discrete=["id"]
    ))
    # id  t1   t2  data1
    # 1   2  90  120    0.1

Acknowledgement
---------------

This implementation originates from an SNCF DTIPG project and is developed and maintained by Mews Labs and SNCF DTIPG.

.. image:: ../../.static/sncf.png
   :alt: SNCF logo
   :width: 100

.. image:: ../../.static/mews_labs.png
   :alt: Mews Labs logo
   :width: 120


.. toctree::
   :maxdepth: 2
   :caption: Contents:

