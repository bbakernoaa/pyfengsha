import datetime
import inspect
from typing import Callable, Any, List, Set, Optional, Dict
import xarray as xr
from .fengsha import dust_emission_fengsha, dust_emission_gocart2g


def _apply_ufunc_wrapper(
    func: Callable,
    ds: xr.Dataset,
    known_core_dims: Set[str],
    output_core_dims: List[List[str]],
    history_message: str,
    core_dims_mapping: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> xr.DataArray:
    """
    Private wrapper to dynamically build and call xr.apply_ufunc.
    This wrapper handles the boilerplate for applying a NumPy-based function
    to the DataArrays within an xarray.Dataset. It dynamically inspects the
    target function's signature to map dataset variables to function arguments.
    A key feature is the ability to handle datasets with non-standard
    dimension names through the `core_dims_mapping` parameter. This allows
    the wrapper to temporarily rename dimensions to a standard internal format
    before the computation and then rename them back on the output,
    preserving the user's original data structure.
    Parameters
    ----------
    func : Callable
        The underlying NumPy function to be wrapped (e.g., `dust_emission_fengsha`).
    ds : xr.Dataset
        The input dataset containing the necessary DataArray variables.
    known_core_dims : Set[str]
        A set of strings representing the *internal* core dimensions that the
        ufunc operates on (e.g., {'lat', 'lon', 'bin'}).
    output_core_dims : List[List[str]]
        A list of lists of strings for the `output_core_dims` argument of
        `apply_ufunc`, using the *internal* dimension names.
    history_message : str
        A message to be added to the output DataArray's history attribute.
    core_dims_mapping : Optional[Dict[str, str]], optional
        A dictionary mapping the user's dimension names to the internal,
        expected dimension names. For example, `{'y': 'lat', 'x': 'lon'}`.
        If None, the function assumes the dataset already uses the
        internal dimension names. Defaults to None.
    **kwargs : Any
        Additional scalar arguments required by `func`.
    Returns
    -------
    xr.DataArray
        The result of the `apply_ufunc` call, with dimensions renamed back
        to the user's original names.
    """
    # Create a new history attribute
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    history = f"{timestamp}: {history_message}"

    # Prepend to existing history if it exists
    if "history" in ds.attrs:
        history = f"{history}\n{ds.attrs['history']}"

    # --- Dimension Renaming ---
    ds_processed = ds
    inverse_mapping = None
    if core_dims_mapping:
        # Only rename dimensions that actually exist in the dataset
        valid_mapping = {
            k: v for k, v in core_dims_mapping.items() if k in ds.dims
        }
        ds_processed = ds.rename(valid_mapping)
        # Create inverse mapping to rename back later
        inverse_mapping = {v: k for k, v in valid_mapping.items()}

    # --- Dynamic Argument Building ---
    sig = inspect.signature(func)
    func_args = []
    for param in sig.parameters.values():
        if param.name in ds_processed:
            func_args.append(ds_processed[param.name])
        elif param.name in kwargs:
            func_args.append(kwargs[param.name])
        else:
            # This provides a more informative error if an argument is missing
            raise ValueError(
                f"Missing required argument for '{func.__name__}': {param.name}"
            )

    # --- Core Dimension Logic (operates on internal names) ---
    input_core_dims = [
        [dim for dim in arg.dims if dim in known_core_dims]
        if isinstance(arg, xr.DataArray)
        else []
        for arg in func_args
    ]

    result = xr.apply_ufunc(
        func,
        *func_args,
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    # Rename dimensions back to the original names if a mapping was used
    if inverse_mapping:
        result = result.rename(inverse_mapping)

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
    and updates the metadata to track data provenance. It can handle
    non-standard dimension names via the `core_dims_mapping` parameter.
    Parameters
    ----------
    ds : xr.Dataset
        An xarray Dataset containing the required variables:
        - fraclake, fracsnow, oro, slc, clay, sand, ssm, rdrag,
        - airdens, ustar, vegfrac, lai, uthrs, distribution
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
        A dictionary mapping the user's dimension names to the internal,
        expected dimension names (e.g., `{'y': 'lat', 'x': 'lon'}`).
        Defaults to None.
    Returns
    -------
    xr.DataArray
        A DataArray containing the calculated dust emission flux. The
        coordinates are preserved from the input Dataset and a history
        attribute is added.
    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> # Create a sample dataset with standard dimension names
    >>> ds = xr.Dataset({
    ...     'fraclake': (('lat', 'lon'), np.zeros((10, 20))),
    ...     'fracsnow': (('lat', 'lon'), np.zeros((10, 20))),
    ...     'oro': (('lat', 'lon'), np.ones((10, 20))),
    ...     'slc': (('lat', 'lon'), np.full((10, 20), 0.2)),
    ...     'clay': (('lat', 'lon'), np.full((10, 20), 0.1)),
    ...     'sand': (('lat', 'lon'), np.full((10, 20), 0.8)),
    ...     'ssm': (('lat', 'lon'), np.full((10, 20), 0.9)),
    ...     'rdrag': (('lat', 'lon'), np.full((10, 20), 0.95)),
    ...     'airdens': (('lat', 'lon'), np.full((10, 20), 1.2)),
    ...     'ustar': (('lat', 'lon'), np.full((10, 20), 0.4)),
    ...     'vegfrac': (('lat', 'lon'), np.full((10, 20), 0.1)),
    ...     'lai': (('lat', 'lon'), np.full((10, 20), 0.2)),
    ...     'uthrs': (('lat', 'lon'), np.full((10, 20), 0.25)),
    ...     'distribution': (('bin',), np.array([0.1, 0.2, 0.7]))
    ... }, coords={'lat': np.arange(10), 'lon': np.arange(20), 'bin': np.arange(3)})
    >>> # Run the FENGSHA model
    >>> emissions = DustEmissionFENGSHA_xr(
    ...     ds=ds.chunk({'lat': 5, 'lon': 10}),
    ...     alpha=1.0, gamma=1.0, kvhmax=2.0E-4, grav=9.81,
    ...     drylimit_factor=1.0, moist_correct=1.0, drag_opt=1
    ... )
    >>> print(emissions.dims)
    ('lat', 'lon', 'bin')
    """
    return _apply_ufunc_wrapper(
        func=dust_emission_fengsha,
        ds=ds,
        known_core_dims={"lat", "lon", "bin"},
        output_core_dims=[["lat", "lon", "bin"]],
        history_message=(
            f"Dust emissions calculated using the FENGSHA scheme (drag_opt={drag_opt})."
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
    contained within a single xarray.Dataset. It preserves input coordinates,
    updates metadata, and can handle non-standard dimension names via the
    `core_dims_mapping` parameter.
    Parameters
    ----------
    ds : xr.Dataset
        An xarray Dataset containing the required variables:
        - radius, fraclake, gwettop, oro, u10m, v10m, du_src
    Ch_DU : float
        Dust emission coefficient.
    grav : float
        Gravity.
    core_dims_mapping : Optional[Dict[str, str]], optional
        A dictionary mapping user dimension names to internal names
        (e.g., `{'y': 'lat', 'x': 'lon'}`). Defaults to None.
    Returns
    -------
    xr.DataArray
        A DataArray containing the calculated dust emission flux. The
        coordinates are preserved from the input Dataset and a history
        attribute is added.
    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> # Create a sample dataset with non-standard dimension names ('y', 'x')
    >>> ds_custom = xr.Dataset({
    ...     'radius': (('particle_size',), np.array([0.1, 0.5, 1.0])),
    ...     'fraclake': (('y', 'x'), np.zeros((10, 20))),
    ...     'gwettop': (('y', 'x'), np.full((10, 20), 0.1)),
    ...     'oro': (('y', 'x'), np.ones((10, 20))),
    ...     'u10m': (('y', 'x'), np.full((10, 20), 5.0)),
    ...     'v10m': (('y', 'x'), np.full((10, 20), 2.0)),
    ...     'du_src': (('y', 'x'), np.ones((10, 20))),
    ... }, coords={'y': np.arange(10), 'x': np.arange(20),
    ...            'particle_size': np.arange(3)})
    >>> # Define the mapping from user's names to internal names
    >>> mapping = {'y': 'lat', 'x': 'lon', 'particle_size': 'bin'}
    >>> # Run the GOCART2G model with the mapping
    >>> emissions = DustEmissionGOCART2G_xr(
    ...     ds=ds_custom.chunk({'y': 5, 'x': 10}),
    ...     Ch_DU=1.0, grav=9.81, core_dims_mapping=mapping
    ... )
    >>> # The output dimensions are the user's original names
    >>> print(emissions.dims)
    ('y', 'x', 'particle_size')
    """
    return _apply_ufunc_wrapper(
        func=dust_emission_gocart2g,
        ds=ds,
        known_core_dims={"lat", "lon", "bin"},
        output_core_dims=[["lat", "lon", "bin"]],
        history_message="Dust emissions calculated using the GOCART2G scheme.",
        core_dims_mapping=core_dims_mapping,
        # Pass scalar arguments to the wrapper via kwargs
        Ch_DU=Ch_DU,
        grav=grav,
    )
