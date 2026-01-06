import datetime
import inspect
from typing import Callable, Any, List, Optional, Dict
import xarray as xr
from .fengsha import dust_emission_fengsha, dust_emission_gocart2g


def _apply_ufunc_wrapper(
    func: Callable,
    ds: xr.Dataset,
    output_core_dims_template: List[List[str]],
    history_message: str,
    core_dims_mapping: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> xr.DataArray:
    """
    Private wrapper to dynamically build and call xr.apply_ufunc.

    Inspects the signature of the wrapped NumPy function (`func`) and
    constructs the arguments for `apply_ufunc` by mapping function
    parameters to variables in the input `xr.Dataset` or to the provided
    `kwargs`. It also allows for user-defined dimension names.

    Parameters
    ----------
    func : Callable
        The underlying NumPy function (e.g., `dust_emission_fengsha`).
    ds : xr.Dataset
        The input dataset.
    output_core_dims_template : List[List[str]]
        A template for the output core dimensions using canonical names
        (e.g., `[['lat', 'lon', 'bin']]`).
    history_message : str
        A message for the output's history attribute.
    core_dims_mapping : Optional[Dict[str, str]], optional
        A dictionary mapping canonical dimension names (e.g., 'lat', 'lon')
        to the actual dimension names in the Dataset (e.g., 'latitude').
        Defaults to `{'lat': 'lat', 'lon': 'lon', 'bin': 'bin'}`.
    **kwargs : Any
        Additional scalar arguments required by `func`.

    Returns
    -------
    xr.DataArray
        The result of the `apply_ufunc` call.
    """
    # --- History Attribute ---
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    history = f"{timestamp}: {history_message}"
    if "history" in ds.attrs:
        history = f"{history}\n{ds.attrs['history']}"

    # --- Dynamic Argument Building ---
    sig = inspect.signature(func)
    func_args = []
    for param in sig.parameters.values():
        if param.name in ds:
            func_args.append(ds[param.name])
        elif param.name in kwargs:
            func_args.append(kwargs[param.name])
        else:
            raise ValueError(
                f"Missing required argument for '{func.__name__}': {param.name}"
            )

    # --- Dimension Handling ---
    if core_dims_mapping is None:
        core_dims_mapping = {"lat": "lat", "lon": "lon", "bin": "bin"}

    known_core_dims = set(core_dims_mapping.values())
    output_core_dims = [
        [core_dims_mapping.get(dim, dim) for dim in dim_list]
        for dim_list in output_core_dims_template
    ]

    input_core_dims = [
        [dim for dim in arg.dims if dim in known_core_dims]
        if isinstance(arg, xr.DataArray)
        else []
        for arg in func_args
    ]

    # --- Ufunc Execution ---
    result = xr.apply_ufunc(
        func,
        *func_args,
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    result.attrs["history"] = history
    return result


def DustEmissionFENGSHA_xr(
    ds: xr.Dataset,
    alpha: float,
    gamma: float,
    kvhmax: float,
    grav: float,
    drylimit_factor: float,
    moist_correct: float,
    drag_opt: int,
    core_dims_mapping: Optional[Dict[str, str]] = None,
) -> xr.DataArray:
    """
    Xarray wrapper for the FENGSHA dust emission scheme.

    This function calculates dust emissions based on a set of input variables
    contained within a single xarray.Dataset. It preserves input coordinates
    and updates the metadata to track data provenance.

    Parameters
    ----------
    ds : xr.Dataset
        An xarray Dataset containing the required variables:
        - fraclake, fracsnow, oro, slc, clay, sand, ssm, rdrag, airdens,
          ustar, vegfrac, lai, uthrs: 2D spatial variables.
        - distribution: 1D feature variable (e.g., by particle size bin).
    alpha : float
        Tuning parameter.
    gamma : float
        Tuning parameter.
    kvhmax : float
        Max KVH ratio.
    grav : float
        Gravity acceleration.
    drylimit_factor : float
        Dry limit factor for moisture correction.
    moist_correct : float
        Moisture correction factor.
    drag_opt : int
        Drag option (1, 2, or 3).
    core_dims_mapping : Optional[Dict[str, str]], optional
        A dictionary mapping canonical dimension names to actual dimension
        names in the Dataset. Defaults to `{'lat': 'lat', 'lon': 'lon', 'bin': 'bin'}`.

    Returns
    -------
    xr.DataArray
        A DataArray containing the calculated dust emission flux, with
        spatial and feature dimensions preserved.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> # Using custom dimension names
    >>> ds = xr.Dataset({
    ...     'fraclake': (('y', 'x'), np.zeros((10, 20))),
    ...     'fracsnow': (('y', 'x'), np.zeros((10, 20))),
    ...     'oro': (('y', 'x'), np.ones((10, 20))),
    ...     'slc': (('y', 'x'), np.full((10, 20), 0.2)),
    ...     'clay': (('y', 'x'), np.full((10, 20), 0.1)),
    ...     'sand': (('y', 'x'), np.full((10, 20), 0.8)),
    ...     'ssm': (('y', 'x'), np.full((10, 20), 0.9)),
    ...     'rdrag': (('y', 'x'), np.full((10, 20), 0.95)),
    ...     'airdens': (('y', 'x'), np.full((10, 20), 1.2)),
    ...     'ustar': (('y', 'x'), np.full((10, 20), 0.4)),
    ...     'vegfrac': (('y', 'x'), np.full((10, 20), 0.1)),
    ...     'lai': (('y', 'x'), np.full((10, 20), 0.2)),
    ...     'uthrs': (('y', 'x'), np.full((10, 20), 0.25)),
    ...     'distribution': (('particle_bin',), np.array([0.1, 0.2, 0.7]))
    ... }, coords={'y': np.arange(10), 'x': np.arange(20), 'particle_bin': np.arange(3)})
    >>> custom_dims = {'lat': 'y', 'lon': 'x', 'bin': 'particle_bin'}
    >>> emissions = DustEmissionFENGSHA_xr(
    ...     ds=ds.chunk({'y': 5, 'x': 10}),
    ...     core_dims_mapping=custom_dims,
    ...     alpha=1.0, gamma=1.0, kvhmax=2.0E-4, grav=9.81,
    ...     drylimit_factor=1.0, moist_correct=1.0, drag_opt=1
    ... )
    >>> print(emissions.dims)
    ('y', 'x', 'particle_bin')
    """
    return _apply_ufunc_wrapper(
        func=dust_emission_fengsha,
        ds=ds,
        output_core_dims_template=[["lat", "lon", "bin"]],
        history_message=(
            f"Dust emissions calculated using FENGSHA scheme (drag_opt={drag_opt})."
        ),
        core_dims_mapping=core_dims_mapping,
        # Pass scalar arguments to the wrapper via kwargs
        alpha=alpha,
        gamma=gamma,
        kvhmax=kvhmax,
        grav=grav,
        drylimit_factor=drylimit_factor,
        moist_correct=moist_correct,
        drag_opt=drag_opt,
    )


def DustEmissionGOCART2G_xr(
    ds: xr.Dataset,
    Ch_DU: float,
    grav: float,
    core_dims_mapping: Optional[Dict[str, str]] = None,
) -> xr.DataArray:
    """
    Xarray wrapper for the GOCART2G dust emission scheme.

    This function calculates dust emissions based on a set of input variables
    contained within a single xarray.Dataset. It preserves input coordinates
    and updates the metadata to track data provenance.

    Parameters
    ----------
    ds : xr.Dataset
        An xarray Dataset containing the following required variables:
        - radius: 1D feature variable (e.g., by particle size bin).
        - fraclake, gwettop, oro, u10m, v10m, du_src: 2D spatial variables.
    Ch_DU : float
        Dust emission coefficient.
    grav : float
        Gravity.
    core_dims_mapping : Optional[Dict[str, str]], optional
        A dictionary mapping canonical dimension names to actual dimension
        names in the Dataset. Defaults to `{'lat': 'lat', 'lon': 'lon', 'bin': 'bin'}`.

    Returns
    -------
    xr.DataArray
        A DataArray containing the calculated dust emission flux, with
        spatial and feature dimensions preserved.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> # Using custom dimension names
    >>> ds = xr.Dataset({
    ...     'radius': (('particle_bin',), np.array([0.1, 0.5, 1.0])),
    ...     'fraclake': (('y', 'x'), np.zeros((10, 20))),
    ...     'gwettop': (('y', 'x'), np.full((10, 20), 0.1)),
    ...     'oro': (('y', 'x'), np.ones((10, 20))),
    ...     'u10m': (('y', 'x'), np.full((10, 20), 5.0)),
    ...     'v10m': (('y', 'x'), np.full((10, 20), 2.0)),
    ...     'du_src': (('y', 'x'), np.ones((10, 20))),
    ... }, coords={'y': np.arange(10), 'x': np.arange(20), 'particle_bin': np.arange(3)})
    >>> custom_dims = {'lat': 'y', 'lon': 'x', 'bin': 'particle_bin'}
    >>> emissions = DustEmissionGOCART2G_xr(
    ...     ds=ds.chunk({'y': 5, 'x': 10}),
    ...     Ch_DU=1.0, grav=9.81, core_dims_mapping=custom_dims
    ... )
    >>> print(emissions.dims)
    ('y', 'x', 'particle_bin')
    """
    return _apply_ufunc_wrapper(
        func=dust_emission_gocart2g,
        ds=ds,
        output_core_dims_template=[["lat", "lon", "bin"]],
        history_message="Dust emissions calculated using the GOCART2G scheme.",
        core_dims_mapping=core_dims_mapping,
        # Pass scalar arguments to the wrapper via kwargs
        Ch_DU=Ch_DU,
        grav=grav,
    )
