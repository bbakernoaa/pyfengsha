import datetime
import inspect
from typing import Callable, Any, List, Set
import xarray as xr
from .fengsha import dust_emission_fengsha, dust_emission_gocart2g


def _apply_ufunc_wrapper(
    func: Callable,
    ds: xr.Dataset,
    known_core_dims: Set[str],
    output_core_dims: List[List[str]],
    history_message: str,
    **kwargs: Any,
) -> xr.DataArray:
    """
    Private wrapper to dynamically build and call xr.apply_ufunc.

    Inspects the signature of the wrapped NumPy function (`func`) and
    constructs the arguments for `apply_ufunc` by mapping function
    parameters to variables in the input `xr.Dataset` or to the provided
    `kwargs`.

    Parameters
    ----------
    func : Callable
        The underlying NumPy function to be wrapped (e.g., `dust_emission_fengsha`).
    ds : xr.Dataset
        The input dataset containing the necessary DataArray variables.
    known_core_dims : Set[str]
        A set of strings representing the core dimensions that the ufunc operates on.
    output_core_dims : List[List[str]]
        A list of lists of strings for the `output_core_dims` argument of `apply_ufunc`.
    history_message : str
        A message to be added to the output DataArray's history attribute.
    **kwargs : Any
        Additional scalar arguments required by the `func`.

    Returns
    -------
    xr.DataArray
        The result of the `apply_ufunc` call.
    """
    # Create a new history attribute
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    history = f"{timestamp}: {history_message}"

    # Prepend to existing history if it exists
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
            # This provides a more informative error if an argument is missing
            raise ValueError(
                f"Missing required argument for '{func.__name__}': {param.name}"
            )

    # Detect spatial dimensions from the first spatial DataArray in the dataset
    spatial_dims = None
    for var_name in ds.data_vars:
        var = ds[var_name]
        if isinstance(var, xr.DataArray) and len(var.dims) >= 2:
            # Look for common spatial dimension patterns
            dims = list(var.dims)
            if any(d in ['lat', 'latitude'] for d in dims) and any(d in ['lon', 'longitude'] for d in dims):
                spatial_dims = [d for d in dims if d in ['lat', 'latitude', 'lon', 'longitude']]
                break

    # If no spatial dims detected, fall back to using the known_core_dims
    if spatial_dims is None:
        spatial_dims = [d for d in known_core_dims if d in ['lat', 'latitude', 'lon', 'longitude']]

    # Update output_core_dims to use detected spatial dimensions
    if len(spatial_dims) >= 2:
        # Assume spatial dims + bin dimension
        if 'bin' in known_core_dims:
            actual_output_dims = spatial_dims + ['bin']
        else:
            actual_output_dims = spatial_dims
        output_core_dims = [actual_output_dims]

    # Dynamically generate input_core_dims by intersecting with all core dims
    all_core_dims = set(spatial_dims) | known_core_dims
    input_core_dims = [
        [dim for dim in arg.dims if dim in all_core_dims]
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
        - fraclake: Fraction of lake coverage
        - fracsnow: Fraction of snow coverage
        - oro: Land/water mask
        - slc: Soil liquid content
        - clay: Clay fraction
        - sand: Sand fraction
        - ssm: Surface soil moisture
        - rdrag: Drag partition parameter
        - airdens: Air density
        - ustar: Friction velocity
        - vegfrac: Vegetation fraction
        - lai: Leaf Area Index
        - uthrs: Threshold velocity
        - distribution: Size distribution per bin
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

    Returns
    -------
    xr.DataArray
        A DataArray containing the calculated dust emission flux, with
        dimensions (lat, lon, bin). The coordinates are preserved from the
        input Dataset and a history attribute is added.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> # Create a sample dataset (replace with real data)
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
    >>> print(emissions.shape)
    (10, 20, 3)
    >>> print('history' in emissions.attrs)
    True
    """
    return _apply_ufunc_wrapper(
        func=dust_emission_fengsha,
        ds=ds,
        known_core_dims={"lat", "lon", "latitude", "longitude", "bin"},
        output_core_dims=[],  # Will be determined dynamically
        history_message=(
            f"Dust emissions calculated using the FENGSHA scheme (drag_opt={drag_opt})."
        ),
        # Pass scalar arguments to the wrapper via kwargs
        alpha=alpha,
        gamma=gamma,
        kvhmax=kvhmax,
        grav=grav,
        drylimit_factor=drylimit_factor,
        moist_correct=moist_correct,
        drag_opt=drag_opt,
    )


def DustEmissionGOCART2G_xr(ds: xr.Dataset, Ch_DU: float, grav: float) -> xr.DataArray:
    """
    Xarray wrapper for the GOCART2G dust emission scheme.

    This function calculates dust emissions based on a set of input variables
    contained within a single xarray.Dataset. It preserves input coordinates
    and updates the metadata to track data provenance.

    Parameters
    ----------
    ds : xr.Dataset
        An xarray Dataset containing the following required variables:
        - radius: Particle radii (bin).
        - fraclake: Fraction of lake coverage (lat, lon).
        - gwettop: Surface wetness (lat, lon).
        - oro: Land mask (lat, lon).
        - u10m: 10m u-wind component (lat, lon).
        - v10m: 10m v-wind component (lat, lon).
        - du_src: Dust source function (lat, lon).
    Ch_DU : float
        Dust emission coefficient.
    grav : float
        Gravity.

    Returns
    -------
    xr.DataArray
        A DataArray containing the calculated dust emission flux, with
        dimensions (lat, lon, bin). The coordinates are preserved from the
        input Dataset and a history attribute is added.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> # Create a sample dataset (replace with real data)
    >>> ds = xr.Dataset({
    ...     'radius': (('bin',), np.array([0.1, 0.5, 1.0])),
    ...     'fraclake': (('lat', 'lon'), np.zeros((10, 20))),
    ...     'gwettop': (('lat', 'lon'), np.full((10, 20), 0.1)),
    ...     'oro': (('lat', 'lon'), np.ones((10, 20))),
    ...     'u10m': (('lat', 'lon'), np.full((10, 20), 5.0)),
    ...     'v10m': (('lat', 'lon'), np.full((10, 20), 2.0)),
    ...     'du_src': (('lat', 'lon'), np.ones((10, 20))),
    ... }, coords={'lat': np.arange(10), 'lon': np.arange(20), 'bin': np.arange(3)})
    >>> # Run the GOCART2G model
    >>> emissions = DustEmissionGOCART2G_xr(
    ...     ds=ds.chunk({'lat': 5, 'lon': 10}),
    ...     Ch_DU=1.0, grav=9.81
    ... )
    >>> print(emissions.shape)
    (10, 20, 3)
    >>> print('history' in emissions.attrs)
    True
    """
    return _apply_ufunc_wrapper(
        func=dust_emission_gocart2g,
        ds=ds,
        known_core_dims={"lat", "lon", "latitude", "longitude", "bin"},
        output_core_dims=[],  # Will be determined dynamically
        history_message="Dust emissions calculated using the GOCART2G scheme.",
        # Pass scalar arguments to the wrapper via kwargs
        Ch_DU=Ch_DU,
        grav=grav,
    )
