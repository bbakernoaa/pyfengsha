import xarray as xr
from .fengsha import (
    dust_emission_fengsha,
    dust_emission_gocart2g
)

def DustEmissionFENGSHA_xr(fraclake: xr.DataArray, fracsnow: xr.DataArray, oro: xr.DataArray, slc: xr.DataArray, clay: xr.DataArray, sand: xr.DataArray,
                           ssm: xr.DataArray, rdrag: xr.DataArray, airdens: xr.DataArray, ustar: xr.DataArray, vegfrac: xr.DataArray, lai: xr.DataArray, uthrs: xr.DataArray,
                           alpha: float, gamma: float, kvhmax: float, grav: float, distribution: xr.DataArray,
                           drylimit_factor: float, moist_correct: float, drag_opt: int) -> xr.DataArray:
    """
    Xarray wrapper for DustEmissionFENGSHA.

    Args:
        fraclake: Fraction of lake coverage (lat, lon).
        fracsnow: Fraction of snow coverage (lat, lon).
        oro: Land/water mask (lat, lon).
        slc: Soil liquid content (lat, lon).
        clay: Clay fraction (lat, lon).
        sand: Sand fraction (lat, lon).
        ssm: Surface soil moisture (lat, lon).
        rdrag: Drag partition parameter (lat, lon).
        airdens: Air density (lat, lon).
        ustar: Friction velocity (lat, lon).
        vegfrac: Vegetation fraction (lat, lon).
        lai: Leaf Area Index (lat, lon).
        uthrs: Threshold velocity (lat, lon).
        alpha: Tuning parameter.
        gamma: Tuning parameter.
        kvhmax: Max KVH ratio.
        grav: Gravity acceleration.
        distribution: Size distribution per bin (bin).
        drylimit_factor: Dry limit factor for moisture correction.
        moist_correct: Moisture correction factor.
        drag_opt: Drag option (1, 2, or 3).

    Returns:
        Dust emission flux (lat, lon, bin).
    """
    return xr.apply_ufunc(
        dust_emission_fengsha,
        fraclake, fracsnow, oro, slc, clay, sand, ssm, rdrag, airdens, ustar, vegfrac, lai, uthrs,
        alpha, gamma, kvhmax, grav, distribution,
        drylimit_factor, moist_correct, drag_opt,
        input_core_dims=[
            ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'],
            ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'],
            [], [], [], [], ['bin'],
            [], [], []
        ],
        output_core_dims=[['lat', 'lon', 'bin']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    )

def DustEmissionGOCART2G_xr(radius: xr.DataArray, fraclake: xr.DataArray, gwettop: xr.DataArray, oro: xr.DataArray, u10m: xr.DataArray, v10m: xr.DataArray, Ch_DU: float, du_src: xr.DataArray, grav: float) -> xr.DataArray:
    """
    Xarray wrapper for DustEmissionGOCART2G.

    Args:
        radius: Particle radii (bin).
        fraclake: Fraction of lake coverage (lat, lon).
        gwettop: Surface wetness (lat, lon).
        oro: Land mask (lat, lon).
        u10m: 10m u-wind component (lat, lon).
        v10m: 10m v-wind component (lat, lon).
        Ch_DU: Dust emission coefficient.
        du_src: Dust source function (lat, lon).
        grav: Gravity.

    Returns:
        Dust emission flux (lat, lon, bin).
    """
    return xr.apply_ufunc(
        dust_emission_gocart2g,
        radius, fraclake, gwettop, oro, u10m, v10m, Ch_DU, du_src, grav,
        input_core_dims=[
            ['bin'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'],
            [], ['lat', 'lon'], []
        ],
        output_core_dims=[['lat', 'lon', 'bin']],
        dask='parallelized',
        output_dtypes=[float]
    )
