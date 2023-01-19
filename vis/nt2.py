#
# Documentation to be added...
#

import xarray as xr
import numpy as np
import multiprocessing as mp


def EdgeToCenter(arr):
    return (arr[1:] + arr[:-1]) / 2


CoordinateDict = {
    "minkowski": {"X1": "x", "X2": "y", "X3": "z"},
    "spherical": {"X1": "r", "X2": "θ", "X3": "φ"},
    "qspherical": {"X1": "r", "X2": "θ", "X3": "φ"},
}


@xr.register_dataarray_accessor("nttplot")
class NTPlotAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def polar(self, ax=None, **kwargs):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        if ax == None:
            ax = plt.gca()
        assert ax.name != "polar", "`ax` must be a rectilinear projection"
        assert len(self._obj.values.shape) == 2, "Data must be 2D"
        ax.grid(False)
        if type(kwargs.get("norm", None)) == mpl.colors.LogNorm:
            cm = kwargs.get("cmap", "viridis")
            cm = plt.get_cmap(cm)
            cm.set_bad(cm(0))
            kwargs["cmap"] = cm
        try:
            r, th = np.meshgrid(self._obj.attrs["re"], self._obj.attrs["θe"])
        except KeyError:
            r, th = np.meshgrid(self._obj.coords["r"], self._obj.coords["θ"])
        except:
            raise KeyError("cell edges must be provided")
        y, x = r * np.cos(th), r * np.sin(th)
        im = ax.pcolormesh(x, y, self._obj.values, rasterized=True, **kwargs)
        ax.set(
            aspect="equal",
            ylabel=kwargs.get("ylabel", "y"),
            xlabel=kwargs.get("xlabel", "x"),
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        if kwargs.get("title", None) == None:
            ax.set_title(f"t={self._obj.coords['t'].values[()]:.2f}")
        else:
            ax.set_title(kwargs.get("title"))
        plt.colorbar(im, cax=cax, label=self._obj.name)


def makeMovie(plot, steps, fpath, dpi=300, num_cpus=mp.cpu_count()):
    from p_tqdm import p_umap
    import matplotlib.pyplot as plt
    import os

    def plotAndSave(plot, st, fpath):
        plot(st)
        plt.savefig(f"{fpath}/{st:05d}.png", dpi=dpi, bbox_inches="tight")
        plt.close()

    if not os.path.exists(fpath):
        os.makedirs(fpath)
    p_umap(lambda st: plotAndSave(plot, st, fpath), steps, num_cpus=num_cpus)


def getFields(fname, steps=None):
    import h5py
    from functools import reduce

    with h5py.File(fname, "r") as f:
        data = {}
        times = []
        nghost = f.attrs["n_ghosts"]
        dimension = f.attrs["dimension"]
        metric = f.attrs["metric"].decode("UTF-8")
        if steps is None:
            steps = range(f.attrs["NumSteps"])
        for step in steps:
            Step = f"Step{step}"
            times.append(f[Step]["time"][()])
            for key in f[Step].keys():
                if key != "step" and key != "time":
                    arr = f[Step][key][nghost:-nghost, nghost:-nghost]
                    coords = list(CoordinateDict[metric].values())[::-1][-dimension:]
                    if key not in data:
                        data[key] = [arr]
                    else:
                        data[key] = np.concatenate((data[key], [arr]), axis=0)
        ds = xr.Dataset(
            {
                reduce(
                    lambda x, y: x.replace(*y),
                    [k.upper(), *list(CoordinateDict[metric].items())],
                ): (
                    ["t", *coords],
                    data[k],
                )
                for k in data.keys()
            },
            coords={
                "t": times,
                **{
                    k: EdgeToCenter(f.attrs[f"x{i+1}"])
                    for i, k in enumerate(coords[::-1])
                },
            },
        )
        for k in ds.data_vars:
            ds[k].attrs = {
                **{k + "e": f.attrs[f"x{i+1}"] for i, k in enumerate(coords[::-1])},
            }
    return ds
