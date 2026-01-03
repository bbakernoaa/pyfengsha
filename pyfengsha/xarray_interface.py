import datetime
import xarray as xr
from .fengsha import dust_emission_fengsha, dust_emission_gocart2g


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
        An xarray Dataset containing the following required variables:
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
    # Create a new history attribute
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    history = (
        f"{timestamp}: Dust emissions calculated using the FENGSHA scheme."
        f" (drag_opt={drag_opt})"
    )

    # Prepend to existing history if it exists
    if "history" in ds.attrs:
        history = f"{history}\n{ds.attrs['history']}"

    # Define the set of possible core dimensions for the underlying NumPy function
    known_core_dims = {"lat", "lon", "bin"}

    # Gather all input arguments for apply_ufunc
    fengsha_args = [
        ds["fraclake"],
        ds["fracsnow"],
        ds["oro"],
        ds["slc"],
        ds["clay"],
        ds["sand"],
        ds["ssm"],
        ds["rdrag"],
        ds["airdens"],
        ds["ustar"],
        ds["vegfrac"],
        ds["lai"],
        ds["uthrs"],
        alpha,
        gamma,
        kvhmax,
        grav,
        ds["distribution"],
        drylimit_factor,
        moist_correct,
        drag_opt,
    ]

    # Dynamically generate input_core_dims by intersecting with known_core_dims
    input_core_dims = [
        [dim for dim in arg.dims if dim in known_core_dims]
        if isinstance(arg, xr.DataArray)
        else []
        for arg in fengsha_args
    ]

    result = xr.apply_ufunc(
        dust_emission_fengsha,
        *fengsha_args,
        input_core_dims=input_core_dims,
        output_core_dims=[["lat", "lon", "bin"]],
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    result.attrs["history"] = history
    return result


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
    # Create a new history attribute
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    history = f"{timestamp}: Dust emissions calculated using the GOCART2G scheme."

    # Prepend to existing history if it exists
    if "history" in ds.attrs:
        history = f"{history}\n{ds.attrs['history']}"

    # Define the set of possible core dimensions
    known_core_dims = {"lat", "lon", "bin"}

    gocart_args = [
        ds["radius"],
        ds["fraclake"],
        ds["gwettop"],
        ds["oro"],
        ds["u10m"],
        ds["v10m"],
        Ch_DU,
        ds["du_src"],
        grav,
    ]

    # Dynamically generate input_core_dims
    input_core_dims = [
        [dim for dim in arg.dims if dim in known_core_dims]
        if isinstance(arg, xr.DataArray)
        else []
        for arg in gocart_args
    ]

    result = xr.apply_ufunc(
        dust_emission_gocart2g,
        *gocart_args,
        input_core_dims=input_core_dims,
        output_core_dims=[["lat", "lon", "bin"]],
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    result.attrs["history"] = history
    return result
