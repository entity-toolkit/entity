import xarray as xr

import nt2.export as exp

useGreek = False
usePickle = False


def configure(use_greek=False, use_pickle=False):
    global useGreek
    global usePickle
    useGreek = use_greek
    usePickle = use_pickle


def DataIs2DPolar(ds):
    return ("r" in ds.dims and ("θ" in ds.dims or "th" in ds.dims)) and len(
        ds.dims
    ) == 2


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

    def fieldplot(
        self,
        fr,
        fth,
        start_points=None,
        sample=None,
        invert_x=False,
        invert_y=False,
        **kwargs,
    ):
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
        invert_x : bool, optional
            Whether to invert the x-axis. Default is False.
        invert_y : bool, optional
            Whether to invert the y-axis. Default is False.
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
        >>> ds.polar.fieldplot("Br", "Bth", sample={"template": "dipole", "nth": 30, "radius": 2.0})
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
            if invert_x:
                fieldline[:, 0] = -fieldline[:, 0]
            if invert_y:
                fieldline[:, 1] = -fieldline[:, 1]
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
        >>> ds.polar.fieldlines("Br", "Bth", [[2.0, np.pi / 4], [2.0, 3 * np.pi / 4]], stopWhen = lambda xy, rth: rth[0] > 5.0)
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
        cell_centered : bool, optional
            Whether the data is cell-centered. Default is True.
        cell_size : float, optional
            If not cell_centered, defines the fraction of the cell to use for coloring. Default is 0.75.
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
        matplotlib.collections.Collection
            The pseudocolor plot.

        Raises
        ------
        AssertionError
            If `ax` is a polar projection or if time is not specified or if data is not 2D polar.

        Notes
        -----
        Additional keyword arguments are passed to `pcolormesh`.
        """

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
        cell_centered = kwargs.pop("cell_centered", True)
        cell_size = kwargs.pop("cell_size", 0.75)

        assert ax.name != "polar", "`ax` must be a rectilinear projection"
        assert "t" not in self._obj.dims, "Time must be specified"
        assert DataIs2DPolar(self._obj), "Data must be 2D polar"
        ax.grid(False)
        if type(kwargs.get("norm", None)) == mpl.colors.LogNorm:
            cm = kwargs.get("cmap", "viridis")
            cm = mpl.colormaps[cm]
            cm.set_bad(cm(0))
            kwargs["cmap"] = cm

        vals = self._obj.values.flatten()
        vals = np.concatenate((vals, vals))
        if not cell_centered:
            drs = self._obj.coords["r_2"] - self._obj.coords["r_1"]
            dths = (
                self._obj.coords["θ_2" if useGreek else "th_2"]
                - self._obj.coords["θ_1" if useGreek else "th_1"]
            )
            r1s = self._obj.coords["r_1"] - drs * cell_size / 2
            r2s = self._obj.coords["r_1"] + drs * cell_size / 2
            th1s = (
                self._obj.coords["θ_1" if useGreek else "th_1"] - dths * cell_size / 2
            )
            th2s = (
                self._obj.coords["θ_1" if useGreek else "th_1"] + dths * cell_size / 2
            )
            rs = np.ravel(np.column_stack((r1s, r2s)))
            ths = np.ravel(np.column_stack((th1s, th2s)))
            nr = len(rs)
            nth = len(ths)
            rs, ths = np.meshgrid(rs, ths)
            rs = rs.flatten()
            ths = ths.flatten()
            points_1 = np.arange(nth * nr).reshape(nth, -1)[:-1:2, :-1:2].flatten()
            points_2 = np.arange(nth * nr).reshape(nth, -1)[:-1:2, 1::2].flatten()
            points_3 = np.arange(nth * nr).reshape(nth, -1)[1::2, 1::2].flatten()
            points_4 = np.arange(nth * nr).reshape(nth, -1)[1::2, :-1:2].flatten()

        else:
            rs = np.append(self._obj.coords["r_1"], self._obj.coords["r_2"][-1])
            ths = np.append(
                self._obj.coords["θ_1" if useGreek else "th_1"],
                self._obj.coords["θ_2" if useGreek else "th_2"][-1],
            )
            nr = len(rs)
            nth = len(ths)
            rs, ths = np.meshgrid(rs, ths)
            rs = rs.flatten()
            ths = ths.flatten()
            points_1 = np.arange(nth * nr).reshape(nth, -1)[:-1, :-1].flatten()
            points_2 = np.arange(nth * nr).reshape(nth, -1)[:-1, 1:].flatten()
            points_3 = np.arange(nth * nr).reshape(nth, -1)[1:, 1:].flatten()
            points_4 = np.arange(nth * nr).reshape(nth, -1)[1:, :-1].flatten()
        x, y = rs * np.sin(ths), rs * np.cos(ths)
        if invert_x:
            x = -x
        if invert_y:
            y = -y
        triang = mpl.tri.Triangulation(
            x,
            y,
            triangles=np.concatenate(
                [
                    np.array([points_1, points_2, points_3]).T,
                    np.array([points_1, points_3, points_4]).T,
                ],
                axis=0,
            ),
        )
        ax.set(
            aspect="equal",
            xlabel=xlabel,
            ylabel=ylabel,
        )
        im = ax.tripcolor(triang, vals, rasterized=True, shading="flat", **kwargs)
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
        return im

    def contour(self, **kwargs):
        """
        Plots a pseudocolor plot of 2D polar data on a rectilinear projection.

        Parameters
        ----------
        ax : Axes object, optional
            The axes on which to plot. Default is the current axes.
        invert_x : bool, optional
            Whether to invert the x-axis. Default is False.
        invert_y : bool, optional
            Whether to invert the y-axis. Default is False.

        Returns
        -------
        matplotlib.contour.QuadContourSet
            The contour plot.

        Raises
        ------
        AssertionError
            If `ax` is a polar projection or if time is not specified or if data is not 2D polar.

        Notes
        -----
        Additional keyword arguments are passed to `contour`.
        """

        import warnings
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        ax = kwargs.pop("ax", plt.gca())
        title = kwargs.pop("title", None)
        invert_x = kwargs.pop("invert_x", False)
        invert_y = kwargs.pop("invert_y", False)

        assert ax.name != "polar", "`ax` must be a rectilinear projection"
        assert "t" not in self._obj.dims, "Time must be specified"
        assert DataIs2DPolar(self._obj), "Data must be 2D polar"
        ax.grid(False)
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
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = ax.contour(x, y, self._obj.values, **kwargs)

        return im


class Metric:
    def __init__(self, base):
        self.base = base


class MinkowskiMetric(Metric):
    def __init__(self):
        super().__init__("minkowski")

    def sqrt_h(self, **coords):
        return 1

    def h_11(self, **coords):
        return 1

    def h_22(self, **coords):
        return 1

    def h_33(self, **coords):
        return 1


class SphericalMetric(Metric):
    def __init__(self):
        super().__init__("spherical")

    def sqrt_h(self, r, th):
        import numpy as np

        return r**2 * np.sin(th)

    def h_11(self, r, th):
        return 1

    def h_22(self, r, th):
        return r**2

    def h_33(self, r, th):
        import numpy as np

        return r**2 * np.sin(th) ** 2


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
    particles: list
        The list of particle species in the simulation. Each element is an Xarray Dataset.

    Methods
    -------
    __del__()
        Closes the HDF5 file.
    __getattr__(name)
        Gets an attribute from the xarray Dataset.
    __getitem__(name)
        Gets an item from the xarray Dataset.

    Examples
    --------
    >>> import nt2.read as nt2r
    >>> data = nt2r.Data("Sim.h5")
    >>> data.Bx.sel(t=10.0, method="nearest").plot()
    """

    def __init__(self, fname):
        if usePickle:
            import h5pickle as h5py
        else:
            import h5py
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
            "cartesian": {"x": "x", "y": "y", "z": "z", "1": "x", "2": "y", "3": "z"},
            "spherical": {
                "r": "r",
                "theta": "θ" if useGreek else "th",
                "phi": "φ" if useGreek else "ph",
                "1": "r",
                "2": "θ" if useGreek else "th",
                "3": "φ" if useGreek else "ph",
            },
        }
        PrtlDict = {
            "cartesian": {
                "X1": "x",
                "X2": "y",
                "X3": "z",
                "U1": "ux",
                "U2": "uy",
                "U3": "uz",
            },
            "spherical": {
                "X1": "r",
                "X2": "θ" if useGreek else "th",
                "X3": "φ" if useGreek else "ph",
                "U1": "ur",
                "U2": "uΘ" if useGreek else "uth",
                "U3": "uφ" if useGreek else "uph",
            },
        }
        self.fname = fname
        try:
            self.file = h5py.File(self.fname, "r")
        except OSError:
            raise OSError(f"Could not open file {self.fname}")
        step0 = list(self.file.keys())[0]
        nsteps = int(self.file.attrs["NumSteps"])
        ngh = int(self.file.attrs["NGhosts"])
        layout = "right" if self.file.attrs["LayoutRight"] == 1 else "left"
        dimension = int(self.file.attrs["Dimension"])
        coordinates = self.file.attrs["Coordinates"].decode("UTF-8")
        if coordinates == "qspherical":
            coordinates = "spherical"
        if coordinates == "spherical":
            self.metric = SphericalMetric()
        else:
            self.metric = MinkowskiMetric()
        coords = list(CoordinateDict[coordinates].values())[::-1][-dimension:]
        # cell-centered coords
        cc_coords = {
            c: EdgeToCenter(self.file.attrs[f"X{i+1}"])
            for i, c in enumerate(coords[::-1])
        }
        # upper and lower limits of the cell
        cell_1 = {
            f"{c}_1": (
                c,
                self.file.attrs[f"X{i+1}"][:-1],
            )
            for i, c in enumerate(coords[::-1])
        }
        cell_2 = {
            f"{c}_2": (
                c,
                self.file.attrs[f"X{i+1}"][1:],
            )
            for i, c in enumerate(coords[::-1])
        }

        times = np.array([self.file[f"Step{s}"]["Time"][()] for s in range(nsteps)])

        if dimension == 1:
            noghosts = slice(ngh, -ngh) if ngh > 0 else slice(None)
        elif dimension == 2:
            noghosts = (slice(ngh, -ngh), slice(ngh, -ngh)) if ngh > 0 else slice(None)
        elif dimension == 3:
            noghosts = (
                (slice(ngh, -ngh), slice(ngh, -ngh), slice(ngh, -ngh))
                if ngh > 0
                else slice(None)
            )

        self.dataset = xr.Dataset()

        fields = [
            k
            for k in self.file[step0].keys()
            if (
                k not in ["Time", "Step"]
                and not (k.startswith("X") or k.startswith("U") or k.startswith("W"))
            )
        ]

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
                [k, *list(CoordinateDict[coordinates].items())],
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
                    **cc_coords,
                    **cell_1,
                    **cell_2,
                },
            )
            self.dataset[k_] = x

        prtls = [
            k
            for k in self.file[step0].keys()
            if (k.startswith("X") or k.startswith("U") or k.startswith("W"))
        ]

        species = np.unique(
            [int(pq.split("_")[1]) for pq in self.file[step0].keys() if pq in prtls]
        )

        def list_to_ragged(arr):
            max_len = np.max([len(a) for a in arr])
            return map(
                lambda a: np.concatenate([a, np.full(max_len - len(a), np.nan)]), arr
            )

        self._particles = {}
        for s in species:
            prtl_data = {}
            for q in [
                f"X1_{s}",
                f"X2_{s}",
                f"X3_{s}",
                f"U1_{s}",
                f"U2_{s}",
                f"U3_{s}",
                f"W_{s}",
            ]:
                if q[0] in ["X", "U"]:
                    q_ = PrtlDict[coordinates][q.split("_")[0]]
                else:
                    q_ = q.split("_")[0]
                if q not in prtls:
                    continue
                if q not in prtl_data.keys():
                    prtl_data[q_] = []
                for k in range(nsteps):
                    step_k = f"Step{k}"
                    if q in self.file[step_k].keys():
                        prtl_data[q_].append(self.file[step_k][q])
                    else:
                        prtl_data[q_].append(np.full_like(prtl_data[q_][-1], np.nan))
                prtl_data[q_] = list_to_ragged(prtl_data[q_])
                prtl_data[q_] = da.from_array(list(prtl_data[q_]))
                prtl_data[q_] = xr.DataArray(
                    prtl_data[q_], dims=["t", "id"], name=q_, coords={"t": times}
                )
            if coordinates == "spherical":
                prtl_data["x"] = (
                    prtl_data[PrtlDict[coordinates]["X1"]]
                    * np.sin(prtl_data[PrtlDict[coordinates]["X2"]])
                    * np.cos(prtl_data[PrtlDict[coordinates]["X3"]])
                )
                prtl_data["y"] = (
                    prtl_data[PrtlDict[coordinates]["X1"]]
                    * np.sin(prtl_data[PrtlDict[coordinates]["X2"]])
                    * np.sin(prtl_data[PrtlDict[coordinates]["X3"]])
                )
                prtl_data["z"] = prtl_data[PrtlDict[coordinates]["X1"]] * np.cos(
                    prtl_data[PrtlDict[coordinates]["X2"]]
                )
            self._particles[s] = xr.Dataset(prtl_data)

    def __del__(self):
        self.file.close()

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, name):
        return self.dataset[name]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()
        self.close()
        for _, v in self._particles.items():
            del v
        del self

    @property
    def particles(self):
        return self._particles

    def plotGrid(self, ax, **kwargs):
        import matplotlib as mpl
        import numpy as np

        coordinates = self.file.attrs["Coordinates"].decode("UTF-8")
        if coordinates == "qspherical":
            coordinates = "spherical"

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        options = {
            "lw": 1,
            "color": "k",
            "ls": "-",
        }
        options.update(kwargs)

        if coordinates == "cartesian":
            for x in self.attrs["X1"]:
                ax.plot([x, x], [self.attrs["X2Min"], self.attrs["X2Max"]], **options)
            for y in self.attrs["X2"]:
                ax.plot([self.attrs["X1Min"], self.attrs["X1Max"]], [y, y], **options)
        else:
            for r in self.attrs["X1"]:
                ax.add_patch(
                    mpl.patches.Arc(
                        (0, 0),
                        2 * r,
                        2 * r,
                        theta1=-90,
                        theta2=90,
                        fill=False,
                        **options,
                    )
                )
            for th in self.attrs["X2"]:
                ax.plot(
                    [
                        self.attrs["X1Min"] * np.sin(th),
                        self.attrs["X1Max"] * np.sin(th),
                    ],
                    [
                        self.attrs["X1Min"] * np.cos(th),
                        self.attrs["X1Max"] * np.cos(th),
                    ],
                    **options,
                )
        ax.set(xlim=xlim, ylim=ylim)

    def makeMovie(self, plot, makeframes=True, **kwargs):
        """
        Makes a movie from a plot function

        Parameters
        ----------
        plot : function
            The plot function to use; accepts output timestep and dataset as arguments.
        makeframes : bool, optional
            Whether to make the frames, or just proceed to making the movie. Default is True.
        num_cpus : int, optional
            The number of CPUs to use for making the frames. Default is None.
        **kwargs :
            Additional keyword arguments passed to `ffmpeg`.
        """
        import numpy as np

        if makeframes:
            makemovie = all(
                exp.makeFrames(
                    plot,
                    np.arange(len(self.t)),
                    f"{self.attrs['Title']}/frames",
                    data=self,
                    num_cpus=kwargs.pop("num_cpus", None),
                )
            )
        else:
            makemovie = True
        if makemovie:
            exp.makeMovie(
                input=f"{self.attrs['Title']}/frames/",
                overwrite=True,
                output=f"{self.attrs['Title']}.mp4",
                number=5,
                **kwargs,
            )
            return True
