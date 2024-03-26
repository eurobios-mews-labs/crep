# Tabular data processing for Continuous REPresentation
[![pytest](https://github.com/eurobios-scb/crep/actions/workflows/pytest.yml/badge.svg?event=push)](https://docs.pytest.org)

```python
pip install git+https://github.com/eurobios-mews-labs/crep
``` 

This simple module aims at providing some function to tackle tabular 
data that have a continuous axis. In situations, this index can represent 
time, but this tool was originally developed to tackle rail way description.

This simple tools helps providing tools to represent a linear structure (cable, rail, beam, pipe) 
whose characteristics are piece-wize constant (of strongly heterogeneous length)

## Basic usage

* **Merge function** merge together two dataframe
```python
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
```
Yield the following result


<img src="examples/basic_example.png" alt="drawing" width="350"/>


## Acknowledgement
This implementation come from an SNCF DTIPG project and is
developed and maintained by Mews Labs and SNCF DTIPG.

<img src="https://www.sncf.com/themes/contrib/sncf_theme/images/logo-sncf.svg?v=3102549095" alt="drawing" width="100"/>
<img src="./.static/mews_labs.png" alt="drawing" width="120"/>
