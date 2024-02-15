import matplotlib.pyplot as plt
import numpy as np
import nt2.read as nt2r

data = nt2r.Data("<PATH-TO>/Langmuir.h5")


def frame(_, data):
    plt.figure(figsize=(5, 10))
    data.Ex.mean("y").plot(yincrease=False)

    omega_p = 1 / data.attrs["units/skindepth0"] / np.sqrt(2)
    T_p = 2 * np.pi / omega_p
    for i in range(10):
        plt.axhline(i * T_p, color="k", ls="--", lw=0.5)
