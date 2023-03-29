#
# Documentation to be added...
#

import xarray as xr

useGreek = False


def configure(use_greek):
    global useGreek
    useGreek = use_greek


def DataIs2DPolar(ds):
    return "r" in ds.dims and ("θ" in ds.dims or "th" in ds.dims) and len(ds.dims) == 2


def DipoleSampling(**kwargs):
    """
    Returns an array of angles sampled from a dipole distribution.

    Parameters
    ----------
    nth : int, optional
        The number of angles to sample. Default is 30.
    pole : float, optional
        The fraction of the angles to sample from the poles. Default is 1/16.

    Returns
    -------
    ndarray
        An array of angles sampled from a dipole distribution.
    """
    import numpy as np

    nth = kwargs.get("nth", 30)
    pole = kwargs.get("pole", 1 / 16)

    nth_poles = int(nth * pole)
    nth_equator = (nth - 2 * nth_poles) // 2
    return np.concatenate(
        [
            np.linspace(0, np.pi * pole, nth_poles + 1)[1:],
            np.linspace(np.pi * pole, np.pi / 2, nth_equator + 2)[1:-1],
            np.linspace(np.pi * (1 - pole), np.pi, nth_poles + 1)[:-1],
        ]
    )


def MonopoleSampling(**kwargs):
    """
    Returns an array of angles sampled from a monopole distribution.

    Parameters
    ----------
    nth : int, optional
        The number of angles to sample. Default is 30.

    Returns
    -------
    ndarray
        An array of angles sampled from a monopole distribution.
    """
    import numpy as np

    nth = kwargs.get("nth", 30)

    return np.linspace(0, np.pi, nth + 2)[1:-1]


@xr.register_dataset_accessor("polar")
class DatasetPolarPlotAccessor:
    import dask

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def pcolor(self, value, **kwargs):
        assert "t" not in self._obj[value].dims, "Time must be specified"
        assert DataIs2DPolar(self._obj), "Data must be 2D polar"
        self._obj[value].polar.pcolor(**kwargs)

    def fieldplot(self, fr, fth, start_points=None, sample=None, **kwargs):
        """
        Plot field lines of a vector field defined by functions fr and fth.

        Parameters
        ----------
        fr : string
            Radial component of the vector field.
        fth : string
            Azimuthal component of the vector field.
        start_points : array_like, optional
            Starting points for the field lines. Either this or `sample` must be specified.
        sample : dict, optional
            Sampling template for generating starting points. Either this or `start_points` must be specified.
            The template can be "dipole" or "monopole". The dict also contains the starting `radius`,
            and the number of points in theta `nth` key.
        **kwargs :
            Additional keyword arguments passed to `fieldlines` and `ax.plot`.

        Raises
        ------
        ValueError
            If neither `start_points` nor `sample` are specified or if an unknown sampling template is given.

        Returns
        -------
        None

        Examples
        --------
        >>> ds.fieldplot("Br", "Bth", sample={"template": "dipole", "nth": 30, "radius": 2.0})
        """
        import matplotlib.pyplot as plt

        if start_points is None and sample is None:
            raise ValueError("Either start_points or sample must be specified")
        elif start_points is None:
            radius = sample.pop("radius", 1.5)
            template = sample.pop("template", "dipole")
            if template == "dipole":
                start_points = [[radius, th] for th in DipoleSampling(**sample)]
            elif template == "monopole":
                start_points = [[radius, th] for th in MonopoleSampling(**sample)]
            else:
                raise ValueError("Unknown sampling template: " + template)

        fieldlines = self.fieldlines(fr, fth, start_points, **kwargs).compute()
        ax = kwargs.pop("ax", plt.gca())
        for fieldline in fieldlines:
            ax.plot(*fieldline.T, **kwargs)

    @dask.delayed
    def fieldlines(self, fr, fth, start_points, **kwargs):
        """
        Compute field lines of a vector field defined by functions fr and fth.

        Parameters
        ----------
        fr : string
            Radial component of the vector field.
        fth : string
            Azimuthal component of the vector field.
        start_points : array_like
            Starting points for the field lines.
        direction : str, optional
            Direction to integrate in. Can be "both", "forward" or "backward". Default is "both".
        stopWhen : callable, optional
            Function that takes the current position and returns True if the integration should stop. Default is to never stop.
        ds : float, optional
            Integration step size. Default is 0.1.
        maxsteps : int, optional
            Maximum number of integration steps. Default is 1000.

        Returns
        -------
        list
            List of field lines.

        Examples
        --------
        >>> ds.fieldlines("Br", "Bth", [[2.0, np.pi / 4], [2.0, 3 * np.pi / 4]], stopWhen = lambda xy, rth: rth[0] > 5.0)
        """

        import numpy as np
        from scipy.interpolate import RegularGridInterpolator

        assert "t" not in self._obj[fr].dims, "Time must be specified"
        assert "t" not in self._obj[fth].dims, "Time must be specified"
        assert DataIs2DPolar(self._obj), "Data must be 2D polar"

        r, th = (
            self._obj.coords["r"].values,
            self._obj.coords["θ" if useGreek else "th"].values,
        )
        _, ths = np.meshgrid(r, th)
        fxs = self._obj[fr] * np.sin(ths) + self._obj[fth] * np.cos(ths)
        fys = self._obj[fr] * np.cos(ths) - self._obj[fth] * np.sin(ths)

        props = dict(method="nearest", bounds_error=False, fill_value=0)
        interpFx = RegularGridInterpolator((th, r), fxs.values, **props)
        interpFy = RegularGridInterpolator((th, r), fys.values, **props)
        return [
            self._fieldline(interpFx, interpFy, rth, **kwargs) for rth in start_points
        ]

    def _fieldline(self, interp_fx, interp_fy, r_th_start, **kwargs):
        import numpy as np
        from copy import copy

        direction = kwargs.pop("direction", "both")
        stopWhen = kwargs.pop("stopWhen", lambda xy, rth: False)
        ds = kwargs.pop("ds", 0.1)
        maxsteps = kwargs.pop("maxsteps", 1000)

        rmax = self._obj.r.max()
        rmin = self._obj.r.min()

        stop = (
            lambda xy, rth: stopWhen(xy, rth)
            or (rth[0] < rmin)
            or (rth[0] > rmax)
            or (rth[1] < 0)
            or (rth[1] > np.pi)
        )

        def integrate(delta, counter):
            r0, th0 = copy(r_th_start)
            XY = np.array([r0 * np.sin(th0), r0 * np.cos(th0)])
            RTH = [r0, th0]
            fieldline = np.array([XY])
            with np.errstate(divide="ignore", invalid="ignore"):
                while range(counter, maxsteps):
                    x, y = XY
                    r = np.sqrt(x**2 + y**2)
                    th = np.arctan2(-y, x) + np.pi / 2
                    RTH = [r, th]
                    vx = interp_fx((th, r))[()]
                    vy = interp_fy((th, r))[()]
                    vmag = np.sqrt(vx**2 + vy**2)
                    XY = XY + delta * np.array([vx, vy]) / vmag
                    if stop(XY, RTH) or np.isnan(XY).any() or np.isinf(XY).any():
                        break
                    else:
                        fieldline = np.append(fieldline, [XY], axis=0)
            return fieldline

        if direction == "forward":
            return integrate(ds, 0)
        elif direction == "backward":
            return integrate(-ds, 0)
        else:
            cntr = 0
            f1 = integrate(ds, cntr)
            f2 = integrate(-ds, cntr)
            return np.append(f2[::-1], f1, axis=0)


@xr.register_dataarray_accessor("polar")
class PolarPlotAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def pcolor(self, **kwargs):
        """
        Plots a pseudocolor plot of 2D polar data on a rectilinear projection.

        Parameters
        ----------
        ax : Axes object, optional
            The axes on which to plot. Default is the current axes.
        cbar_size : str, optional
            The size of the colorbar. Default is "5%".
        cbar_pad : float, optional
            The padding between the colorbar and the plot. Default is 0.05.
        cbar_position : str, optional
            The position of the colorbar. Default is "right".
        cbar_ticksize : int or float, optional
            The size of the ticks on the colorbar. Default is None.
        title : str, optional
            The title of the plot. Default is None.
        invert_x : bool, optional
            Whether to invert the x-axis. Default is False.
        invert_y : bool, optional
            Whether to invert the y-axis. Default is False.
        ylabel : str, optional
            The label for the y-axis. Default is "y".
        xlabel : str, optional
            The label for the x-axis. Default is "x".
        label : str, optional
            The label for the plot. Default is None.

        Returns
        -------
        QuadMesh
            A `QuadMesh` object representing the pseudocolor plot.

        Raises
        ------
        AssertionError
            If `ax` is a polar projection or if time is not specified or if data is not 2D polar.

        Notes
        -----
        Additional keyword arguments are passed to `pcolormesh`.
        """

        import warnings
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        ax = kwargs.pop("ax", plt.gca())
        cbar_size = kwargs.pop("cbar_size", "5%")
        cbar_pad = kwargs.pop("cbar_pad", 0.05)
        cbar_pos = kwargs.pop("cbar_position", "right")
        cbar_orientation = (
            "vertical" if cbar_pos == "right" or cbar_pos == "left" else "horizontal"
        )
        cbar_ticksize = kwargs.pop("cbar_ticksize", None)
        title = kwargs.pop("title", None)
        invert_x = kwargs.pop("invert_x", False)
        invert_y = kwargs.pop("invert_y", False)
        ylabel = kwargs.pop("ylabel", "y")
        xlabel = kwargs.pop("xlabel", "x")
        label = kwargs.pop("label", None)

        assert ax.name != "polar", "`ax` must be a rectilinear projection"
        assert "t" not in self._obj.dims, "Time must be specified"
        assert DataIs2DPolar(self._obj), "Data must be 2D polar"
        ax.grid(False)
        if type(kwargs.get("norm", None)) == mpl.colors.LogNorm:
            cm = kwargs.get("cmap", "viridis")
            cm = mpl.colormaps[cm]
            cm.set_bad(cm(0))
            kwargs["cmap"] = cm
        r, th = np.meshgrid(
            self._obj.coords["r"], self._obj.coords["θ" if useGreek else "th"]
        )
        x, y = r * np.sin(th), r * np.cos(th)
        if invert_x:
            x = -x
        if invert_y:
            y = -y
        ax.set(
            aspect="equal",
            xlabel=xlabel,
            ylabel=ylabel,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = ax.pcolormesh(x, y, self._obj.values, rasterized=True, **kwargs)
        if cbar_pos is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(cbar_pos, size=cbar_size, pad=cbar_pad)
            _ = plt.colorbar(
                im,
                cax=cax,
                label=self._obj.name if label is None else label,
                orientation=cbar_orientation,
            )
            if cbar_orientation == "vertical":
                axis = cax.yaxis
            else:
                axis = cax.xaxis
            axis.set_label_position(cbar_pos)
            axis.set_ticks_position(cbar_pos)
            if cbar_ticksize is not None:
                cax.tick_params("both", labelsize=cbar_ticksize)
        ax.set_title(
            f"t={self._obj.coords['t'].values[()]:.2f}" if title is None else title
        )


class Data:
    """
    A class to load data from the Entity single-HDF5 file and store it as a lazily loaded xarray Dataset.

    Parameters
    ----------
    fname : str
        The name of the HDF5 file to read.

    Attributes
    ----------
    fname : str
        The name of the HDF5 file.
    file : h5py.File
        The HDF5 file object.
    dataset : xr.Dataset
        The xarray Dataset containing the loaded data.

    Methods
    -------
    __del__()
        Closes the HDF5 file.
    __getattr__(name)
        Gets an attribute from the xarray Dataset.
    __getitem__(name)
        Gets an item from the xarray Dataset.
    """

    def __init__(self, fname):
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
        self.fname = fname
        self.file = h5py.File(self.fname, "r")
        step0 = list(self.file.keys())[0]
        nsteps = self.file.attrs["NumSteps"]
        ngh = self.file.attrs["NGhosts"]
        layout = "right" if self.file.attrs["LayoutRight"] == 1 else "left"
        dimension = self.file.attrs["Dimension"]
        metric = self.file.attrs["Metric"].decode("UTF-8")
        coords = list(CoordinateDict[metric].values())[::-1][-dimension:]
        times = np.array([self.file[f"Step{s}"]["Time"][()] for s in range(nsteps)])

        if dimension == 1:
            noghosts = slice(ngh, -ngh)
        elif dimension == 2:
            noghosts = (slice(ngh, -ngh), slice(ngh, -ngh))
        elif dimension == 3:
            noghosts = (slice(ngh, -ngh), slice(ngh, -ngh), slice(ngh, -ngh))

        self.dataset = xr.Dataset()

        fields = [k for k in self.file[step0].keys() if k not in ["Time", "Step"]]

        for k in self.file.attrs.keys():
            if (
                type(self.file.attrs[k]) == bytes
                or type(self.file.attrs[k]) == np.bytes_
            ):
                self.dataset.attrs[k] = self.file.attrs[k].decode("UTF-8")
            else:
                self.dataset.attrs[k] = self.file.attrs[k]

        for k in fields:
            dask_arrays = []
            for s in range(nsteps):
                array = da.from_array(
                    np.transpose(self.file[f"Step{s}/{k}"])
                    if layout == "right"
                    else self.file[f"Step{s}/{k}"]
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
                        k: EdgeToCenter(self.file.attrs[f"X{i+1}"])
                        for i, k in enumerate(coords[::-1])
                    },
                },
            )
            self.dataset[k_] = x

    def __del__(self):
        self.file.close()

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, name):
        return self.dataset[name]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self


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
                  extension="png", compression="1", overwrite=True, output="anim.mov")
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
        "-y" if ffmpeg_kwargs.get("overwrite", False) else None,
        ffmpeg_kwargs.get("output", "anim.mov"),
    ]
    command = [str(c) for c in command if c is not None]
    print("Command:\n", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print("ffmpeg -- [OK]")
        return True
    else:
        print("ffmpeg -- [not OK]", result.returncode, result.stdout, result.stderr)
        return False


def makeFrames(plot, steps, fpath, num_cpus=None):
    """
    Create plot frames from a set of timesteps of the same dataset.
    Parameters
    ----------
    plot : function
        A function that generates and saves the plot. The function must take a time index
        as an argument.
    steps : array_like, optional
        The time indices to use for generating the movie.
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
    >>> makeFrames(plot_func, range(100), 'output/', num_cpus=16)
    """

    from tqdm import tqdm
    import multiprocessing as mp
    import matplotlib.pyplot as plt
    import os

    global plotAndSave

    def plotAndSave(ti, t, fpath):
        try:
            plot(t)
            plt.savefig(f"{fpath}/{ti:05d}.png")
            plt.close()
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    if num_cpus is None:
        num_cpus = mp.cpu_count()

    pool = mp.Pool(num_cpus)

    # if fpath doesn't exist, create it
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    tasks = [[ti, t, fpath] for ti, t in enumerate(steps)]
    results = [pool.apply_async(plotAndSave, t) for t in tasks]
    return [result.get() for result in tqdm(results)]
