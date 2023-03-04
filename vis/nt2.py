#
# Documentation to be added...
#

import xarray as xr
import multiprocessing as mp

useGreek = False


def configure(use_greek):
    global useGreek
    useGreek = use_greek


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
            cm = mpl.colormaps[cm]
            cm.set_bad(cm(0))
            kwargs["cmap"] = cm
        r, th = np.meshgrid(
            self._obj.coords["r"], self._obj.coords["θ" if useGreek else "th"]
        )
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

    QuantityDict = {
        "Ttt": "E",
        "Ttx": "Px",
        "Tty": "Py",
        "Ttz": "Pz",
    }

    CoordinateDict = {
        "minkowski": {"x": "x", "y": "y", "z": "z", "1": "x", "2": "y", "3": "z"},
        "spherical": {
            "r": "r",
            "theta": "θ" if useGreek else "th",
            "phi": "φ" if useGreek else "ph",
            "1": "r",
            "2": "θ" if useGreek else "th",
            "3": "φ" if useGreek else "ph",
        },
        "qspherical": {
            "r": "r",
            "theta": "θ" if useGreek else "th",
            "phi": "φ" if useGreek else "ph",
            "1": "r",
            "2": "θ" if useGreek else "th",
            "3": "φ" if useGreek else "ph",
        },
    }

    file = h5py.File(fname, "r")
    step0 = list(file.keys())[0]
    nsteps = file.attrs["NumSteps"]
    ngh = file.attrs["NGhosts"]
    layout = "right" if file.attrs["LayoutRight"] == 1 else "left"
    dimension = file.attrs["Dimension"]
    metric = file.attrs["Metric"].decode("UTF-8")
    coords = list(CoordinateDict[metric].values())[::-1][-dimension:]
    times = np.array([file[f"Step{s}"]["Time"][()] for s in range(nsteps)])

    if dimension == 1:
        noghosts = slice(ngh, -ngh)
    elif dimension == 2:
        noghosts = (slice(ngh, -ngh), slice(ngh, -ngh))
    elif dimension == 3:
        noghosts = (slice(ngh, -ngh), slice(ngh, -ngh), slice(ngh, -ngh))

    ds = xr.Dataset()

    fields = [k for k in file[step0].keys() if k not in ["Time", "Step"]]

    for k in file.attrs.keys():
        if type(file.attrs[k]) == bytes or type(file.attrs[k]) == np.bytes_:
            ds.attrs[k] = file.attrs[k].decode("UTF-8")
        else:
            ds.attrs[k] = file.attrs[k]

    for k in fields:
        dask_arrays = []
        for s in range(nsteps):
            array = da.from_array(
                np.transpose(file[f"Step{s}/{k}"])
                if layout == "right"
                else file[f"Step{s}/{k}"]
            )
            dask_arrays.append(array[noghosts])

        k_ = reduce(
            lambda x, y: x.replace(*y)
            if "_" not in x
            else "_".join([x.split("_")[0].replace(*y)] + x.split("_")[1:]),
            [k, *list(CoordinateDict[metric].items())],
        )
        k_ = reduce(
            lambda x, y: x.replace(*y),
            [k_, *list(QuantityDict.items())],
        )
        x = xr.DataArray(
            da.stack(dask_arrays, axis=0),
            dims=["t", *coords],
            name=k_,
            coords={
                "t": times,
                **{
                    k: EdgeToCenter(file.attrs[f"X{i+1}"])
                    for i, k in enumerate(coords[::-1])
                },
            },
        )
        ds[k_] = x
    return ds


def makeMovie(**ffmpeg_kwargs):
    """
    Create a movie from frames using the `ffmpeg` command-line tool.

    Parameters
    ----------
    ffmpeg_kwargs : dict
        Keyword arguments for the `ffmpeg` command-line tool.

    Returns
    -------
    bool
        True if the movie was created successfully, False otherwise.

    Notes
    -----
    This function uses the `subprocess` module to execute the `ffmpeg` command-line
    tool with the given arguments.

    Examples
    --------
    >>> makeMovie(ffmpeg="/path/to/ffmpeg", framerate="30", start="0", input="step_", number=3,
                  extension="png", compression="1", output="anim.mov")
    """
    import subprocess

    command = [
        ffmpeg_kwargs.get("ffmpeg", "ffmpeg"),
        "-nostdin",
        "-framerate",
        ffmpeg_kwargs.get("framerate", "30"),
        "-start_number",
        ffmpeg_kwargs.get("start", "0"),
        "-i",
        ffmpeg_kwargs.get("input", "step_")
        + f"%0{ffmpeg_kwargs.get('number', 3)}d.{ffmpeg_kwargs.get('extension', 'png')}",
        "-c:v",
        "libx264",
        "-crf",
        ffmpeg_kwargs.get("compression", "1"),
        "-filter_complex",
        "[0:v]format=yuv420p,pad=ceil(iw/2)*2:ceil(ih/2)*2",
        ffmpeg_kwargs.get("output", "anim.mov"),
    ]
    command = [str(c) for c in command]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print("ffmpeg -- [OK]")
        return True
    else:
        print("ffmpeg -- [not OK]", result.returncode, result.stdout, result.stderr)
        return False


def makeFrames(plot, ds, steps, fpath, num_cpus=mp.cpu_count()):
    """
    Create plot frames from a set of timesteps of the same dataset.

    Parameters
    ----------
    plot : function
        A function that generates and saves the plot. The function must take a time index
        as its first argument, followed by the data source, and the path for png files.
    ds : object
        The data source used to generate the plots.
    steps : array_like, optional
        The time indices to use for generating the movie. If None, use all
        time indices in `ds` (defined by the `t` key).
    fpath : str
        The file path to save the frames.
    num_cpus : int, optional
        The number of CPUs to use for parallel processing. If None, use all available CPUs.

    Returns
    -------
    list
        A list of results returned by the `plot` function, one for each time index.

    Raises
    ------
    ValueError
        If `plot` is not a callable function.

    Notes
    -----
    This function uses the `multiprocessing` module to parallelize the generation
    of the plots, and `tqdm` module to display a progress bar.

    Examples
    --------
    >>> makeFrames(plot_func, data_source, range(100), 'output/', num_cpus=16)
    """

    from tqdm import tqdm
    import numpy as np
    import multiprocessing as mp

    if num_cpus is None:
        num_cpus = mp.cpu_count()

    pool = mp.Pool(num_cpus)

    if steps is None:
        steps = np.arange(len(ds.t))
    else:
        steps = np.array(steps)

    tasks = [[ti, ds, fpath] for ti in steps]
    results = [pool.apply_async(plot, t) for t in tasks]
    return [result.get() for result in tqdm(results)]
