#
# Documentation to be added...
#

import xarray as xr
import multiprocessing as mp


@xr.register_dataarray_accessor("nttplot")
class NTPlotAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def polar(self, ax=None, **kwargs):
        import warnings
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from mpl_toolkits.axes_grid1 import make_axes_locatable

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
        r, th = np.meshgrid(self._obj.coords["r"], self._obj.coords["θ"])
        y, x = r * np.cos(th), r * np.sin(th)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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


def getFields(fname):
    """
    Reads in a hdf5 file at `fname`, extracts the fields and metadata, and returns them in a lazy xarray dataset.

    Parameters
    ----------
    fname : str
        The file name of the hdf5 file.

    Returns
    -------
    xarray.Dataset
        A lazy-loaded xarray dataset containing the fields and metadata read from the hdf5 file.
    """
    import h5pickle as h5py
    import dask.array as da
    from functools import reduce
    import numpy as np

    def EdgeToCenter(arr):
        return (arr[1:] + arr[:-1]) / 2

    CoordinateDict = {
        "minkowski": {"x": "x", "y": "y", "z": "z"},
        "spherical": {"r": "r", "theta": "θ", "phi": "φ"},
        "qspherical": {"r": "r", "theta": "θ", "phi": "φ"},
    }

    file = h5py.File(fname, "r")
    step0 = list(file.keys())[0]
    nsteps = file.attrs["NumSteps"]
    ngh = file.attrs["n_ghosts"]
    dimension = file.attrs["dimension"]
    metric = file.attrs["metric"].decode("UTF-8")
    coords = list(CoordinateDict[metric].values())[::-1][-dimension:]

    ds = xr.Dataset()

    fields = [k for k in file[step0].keys() if k not in ["time", "step"]]

    for k in file.attrs.keys():
        if type(file.attrs[k]) == bytes or type(file.attrs[k]) == np.bytes_:
            ds.attrs[k] = file.attrs[k].decode("UTF-8")
        else:
            ds.attrs[k] = file.attrs[k]

    for k in fields:
        dask_arrays = []
        times = []
        for s in range(nsteps):
            array = da.from_array(file[f"Step{s}/{k}"])
            times.append(file[f"Step{s}"]["time"][()])
            dask_arrays.append(array[ngh:-ngh, ngh:-ngh])

        k_ = reduce(
            lambda x, y: x.replace(*y),
            [k, *list(CoordinateDict[metric].items())],
        )
        x = xr.DataArray(
            da.stack(dask_arrays, axis=0),
            dims=["t", *coords],
            name=k_,
            coords={
                "t": times,
                **{
                    k: EdgeToCenter(file.attrs[f"x{i+1}"])
                    for i, k in enumerate(coords[::-1])
                },
            },
        )
        ds[k_] = x
    return ds


def plotAndSave(st, plot_, fpath_, dpi_):
    import matplotlib.pyplot as plt

    try:
        plot_(st)
        plt.savefig(f"{fpath_}/{st:05d}.png", dpi=dpi_, bbox_inches="tight")
        plt.close()
        return True
    except:
        return False


def makeMovie(plot, steps, fpath, dpi=300, num_cpus=mp.cpu_count()):
    """
    Generates a movie by applying the `plot` function to each step in `steps`, and saving the resulting figure at the specified `fpath` with the specified `dpi`.

    Parameters
    ----------
    plot : function
        A function that accepts a single argument 'st' and returns a figure.
    steps : list
        A list of steps that the `plot` function will be applied to.
    fpath : str
        The file path where the figures will be saved.
    dpi : int, optional
        The dpi of the saved figures. Defaults to 300.
    num_cpus : int, optional
        The number of CPUs to use for parallel processing. Defaults to the number of CPUs on the current machine.

    Returns
    -------
    None
    """
    import tqdm
    from functools import partial

    # from multiprocessing import Pool
    import os
    from multiprocessing import get_context

    print(f"Processing on {num_cpus} CPUs\n")
    print(f"1. Writing frames to `{fpath}/%05d.png`.")

    if not os.path.exists(fpath):
        os.makedirs(fpath)

    # with Pool(num_cpus) as p:
    with get_context("spawn").Pool(num_cpus) as pool:
        with tqdm.tqdm(total=len(steps)) as pbar:
            for _ in pool.imap_unordered(
                partial(plotAndSave, plot_=plot, fpath_=fpath, dpi_=dpi), steps
            ):
                pbar.update()

    #     result = list(
    #         tqdm.tqdm(
    #             p.imap_unordered(
    #                 partial(plotAndSave, plot_=plot, fpath_=fpath, dpi_=dpi),
    #                 steps,
    #             ),
    #             total=len(steps),
    #         )
    #     )
    # return result
