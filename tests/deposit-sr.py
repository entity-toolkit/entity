import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

prtl = pd.read_csv("../build/tests/prtl.csv")
ex1 = pd.read_csv("../build/tests/ex1.csv")
ex2 = pd.read_csv("../build/tests/ex2.csv")
metrics = pd.read_csv("../build/tests/metric.csv")
q = pd.read_csv("../build/tests/q.csv")

prtl["x1"] = prtl["i1"] + prtl["dx1"]
prtl["x2"] = prtl["i2"] + prtl["dx2"]

# plot 1
plt.scatter(
    prtl["x1"],
    prtl["x2"],
    c=mpl.colormaps["turbo"](np.linspace(0, 1, len(prtl))),
)
plt.xlim(127.5, 130.5)
plt.ylim(127.5, 129.5)
plt.axvline(128, color="k")
plt.axhline(128, color="k")
plt.axvline(129, color="k")
plt.axhline(129, color="k")
plt.axvline(130, color="k")
plt.gca().set_aspect("equal")

# plot 2
q_divA = (
    ex1["ex1_{i+1/2;j}"] * metrics["h_{i+1/2;j}"].to_numpy()
    - ex1["ex1_{i-1/2;j}"] * metrics["h_{i-1/2;j}"].to_numpy()
) + (
    ex2["ex2_{i;j+1/2}"] * metrics["h_{i;j+1/2}"].to_numpy()
    - ex2["ex2_{i;j-1/2}"] * metrics["h_{i;j-1/2}"].to_numpy()
)
q_divB = (
    ex1["ex1_{i+3/2;j}"] * metrics["h_{i+3/2;j}"].to_numpy()
    - ex1["ex1_{i+1/2;j}"] * metrics["h_{i+1/2;j}"].to_numpy()
) + (
    ex2["ex2_{i+1;j+1/2}"] * metrics["h_{i+1;j+1/2}"].to_numpy()
    - ex2["ex2_{i+1;j-1/2}"] * metrics["h_{i+1;j-1/2}"].to_numpy()
)
q_divC = (
    ex1["ex1_{i+5/2;j}"] * metrics["h_{i+5/2;j}"].to_numpy()
    - ex1["ex1_{i+3/2;j}"] * metrics["h_{i+3/2;j}"].to_numpy()
) + (
    ex2["ex2_{i+2;j+1/2}"] * metrics["h_{i+2;j+1/2}"].to_numpy()
    - ex2["ex2_{i+2;j-1/2}"] * metrics["h_{i+2;j-1/2}"].to_numpy()
)
q_divD = (
    ex1["ex1_{i+1/2;j+1}"] * metrics["h_{i+1/2;j+1}"].to_numpy()
    - ex1["ex1_{i-1/2;j+1}"] * metrics["h_{i-1/2;j+1}"].to_numpy()
) + (
    ex2["ex2_{i;j+3/2}"] * metrics["h_{i;j+3/2}"].to_numpy()
    - ex2["ex2_{i;j+1/2}"] * metrics["h_{i;j+1/2}"].to_numpy()
)
q_divE = (
    ex1["ex1_{i+3/2;j+1}"] * metrics["h_{i+3/2;j+1}"].to_numpy()
    - ex1["ex1_{i+1/2;j+1}"] * metrics["h_{i+1/2;j+1}"].to_numpy()
) + (
    ex2["ex2_{i+1;j+3/2}"] * metrics["h_{i+1;j+3/2}"].to_numpy()
    - ex2["ex2_{i+1;j+1/2}"] * metrics["h_{i+1;j+1/2}"].to_numpy()
)
q_divF = (
    ex1["ex1_{i+5/2;j+1}"] * metrics["h_{i+5/2;j+1}"].to_numpy()
    - ex1["ex1_{i+3/2;j+1}"] * metrics["h_{i+3/2;j+1}"].to_numpy()
) + (
    ex2["ex2_{i+2;j+3/2}"] * metrics["h_{i+2;j+3/2}"].to_numpy()
    - ex2["ex2_{i+2;j+1/2}"] * metrics["h_{i+2;j+1/2}"].to_numpy()
)

labels = ["{i,j}", "{i+1,j}", "{i+2,j}", "{i,j+1}", "{i+1,j+1}", "{i+2,j+1}"]
qs = [q[f"q_{K}"] for K in ["A", "B", "C", "D", "E", "F"]]
for l, q in zip(labels, qs):
    plt.plot(q, label=rf"$q_{{{l}}}$")

plt.plot(np.sum(qs, axis=0), label=r"$\sum q$", lw=2, c="r")
plt.legend()
plt.axhline(0, c="gray", ls="--")

# plot 3
plt.plot(
    100 * (np.sum(qs, axis=0)) / np.max(qs),
    label=r"$100\%\times\sum q / max(q)$",
)
plt.ylabel("% error")
plt.legend()
plt.axhline(0, c="gray", ls="--")

# plot 4
labels = ["{i,j}", "{i+1,j}", "{i+2,j}", "{i,j+1}", "{i+1,j+1}", "{i+2,j+1}"]
qs = [q_divA, q_divB, q_divC, q_divD, q_divE, q_divF]
for l, q in zip(labels, qs):
    plt.plot(q, label=rf"$q_{{{l}}}$")

plt.plot(np.sum(qs, axis=0), label=r"$\sum q$", lw=2, c="r")
plt.legend()
plt.axhline(0, c="gray", ls="--")

# plot 5
plt.plot(
    100 * (np.sum(qs, axis=0)) / np.max(qs),
    label=r"$100\%\times\sum q / max(q)$",
)
plt.ylabel("% error")
plt.legend()
plt.axhline(0, c="gray", ls="--")
