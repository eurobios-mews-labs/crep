# Copyright 2023 Eurobios
# Licensed under the CeCILL License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://cecill.info/
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler

from crep import merge

matplotlib.use("qt5agg")
plt.style.use('./examples/.matplotlibrc')
plt.rcParams['figure.figsize'] = [5.0, 5.0]
plt.rcParams['figure.dpi'] = 200
n = 8
default_cycler = cycler('color',
                        ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97',
                         '#474747', '#9e9e9e'])

plt.style.use("bmh")
plt.rcParams["font.family"] = "ubuntu"
plt.rcParams['axes.facecolor'] = "white"
plt.rcParams['axes.prop_cycle'] = default_cycler

cmap = plt.get_cmap("rainbow_r")
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
eta_cmap = "rocket_r"


def plot(data: pd.DataFrame, continuous_index, data_col, color="C1",
         legend=None):
    id1, id2 = continuous_index
    for i, r in data.iterrows():
        label = legend if i == 0 else None
        plt.plot([r[id1], r[id2]], [r[data_col], r[data_col]], marker=".",
                 color=color, label=label)


df_left = pd.DataFrame(
    dict(id=[2, 2, 2],
         t1=[0, 100, 120],
         t2=[100, 120, 130],
         data1=[0.2, 0.1, 0.3])
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

_, axes = plt.subplots(nrows=2, sharex=True)

ret = ret[ret["id"] == 2]
plt.sca(axes[0])
plot(df_left, continuous_index=["t1", "t2"], data_col="data1", color="C1",
     legend='data 1')
plot(df_right, continuous_index=["t1", "t2"], data_col="data2", color="C2",
     legend='data 2')
plt.ylabel("raw data")
plt.legend()
plt.sca(axes[1])
plot(ret, continuous_index=["t1", "t2"], data_col="data1", color="C1")
plot(ret, continuous_index=["t1", "t2"], data_col="data2", color="C2")
plt.xlabel("$t$")
plt.ylabel("merged data")
plt.legend()
plt.savefig("examples/basic_example.png")
