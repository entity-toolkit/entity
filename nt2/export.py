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
                  extension="png", compression="1", overwrite=True, output="anim.mp4")
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
        ffmpeg_kwargs.get("output", "anim.mp4"),
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


def makeFrames(plot, steps, fpath, data=None, num_cpus=None):
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
    data : xarray.Dataset, optional
        The dataset to use for generating the movie (passed to plot as the second argument)
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
            if data is None:
                plot(t)
            else:
                plot(t, data)
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
