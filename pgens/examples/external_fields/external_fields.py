import nt2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = True

data = nt2.Data("external_fields")
prtls = data.particles.load()

fig = plt.figure(figsize=(10, 5), dpi=150)
ax = fig.add_subplot(111)

ax.scatter(
    prtls["x"],
    prtls["y"],
    s=1,
    color=np.choose(prtls["sp"].array - 1, ["r", "g", "b"]),
    ec=None,
    alpha=(100 - prtls["t"].array) / 100,
)
ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel=r"$x$", ylabel=r"$y$", aspect=1)

plt.show()
