try:
    import xarray as xr
    import numpy as np
    from .fengsha import (
        DustEmissionFENGSHA,
        DustEmissionGOCART2G,
        DustEmissionK14
    )
except ImportError:
    pass

def DustEmissionFENGSHA_xr(fraclake, fracsnow, oro, slc, clay, sand, silt,
                           ssm, rdrag, airdens, ustar, vegfrac, lai, uthrs,
                           alpha, gamma, kvhmax, grav, rhop, distribution,
                           drylimit_factor, moist_correct, drag_opt):
    """
    Xarray wrapper for DustEmissionFENGSHA.
    """
    return xr.apply_ufunc(
        DustEmissionFENGSHA,
        fraclake, fracsnow, oro, slc, clay, sand, silt,
        ssm, rdrag, airdens, ustar, vegfrac, lai, uthrs,
        alpha, gamma, kvhmax, grav, rhop, distribution,
        drylimit_factor, moist_correct, drag_opt,
        input_core_dims=[
            ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'],
            ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'],
            [], [], [], [], ['bin'], ['bin'],
            [], [], []
        ],
        output_core_dims=[['lat', 'lon', 'bin']],
        vectorize=True, # Numba function expects 2D/1D arrays, so we can't let apply_ufunc loop over core dims implicitly unless we use dask="parallelized" or vectorize=True if inputs have extra dims.
                        # Wait, input_core_dims means these are passed as whole arrays to the function.
                        # The function DustEmissionFENGSHA expects 2D arrays (lat, lon) and 1D arrays (bin).
                        # So vectorize=False is correct if inputs exactly match core dims + broadcast dims.
                        # However, apply_ufunc without dask="parallelized" passes numpy arrays.
                        # If we have extra dims (time), apply_ufunc loops over them.
                        # So vectorize=False should work and be efficient.
        dask='parallelized',
        output_dtypes=[float]
    )

def DustEmissionGOCART2G_xr(radius, fraclake, gwettop, oro, u10m, v10m, Ch_DU, du_src, grav):
    """
    Xarray wrapper for DustEmissionGOCART2G.
    """
    return xr.apply_ufunc(
        DustEmissionGOCART2G,
        radius, fraclake, gwettop, oro, u10m, v10m, Ch_DU, du_src, grav,
        input_core_dims=[
            ['bin'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'],
            [], ['lat', 'lon'], []
        ],
        output_core_dims=[['lat', 'lon', 'bin']],
        dask='parallelized',
        output_dtypes=[float]
    )

def DustEmissionK14_xr(t_soil, w_top, rho_air, z0, z, u_z, v_z, ustar,
                       f_land, f_snow, f_src, f_sand, f_silt, f_clay,
                       texture, vegetation, gvf,
                       f_w, f_c, uts_gamma, UNDEF, GRAV, VON_KARMAN,
                       opt_clay, Ch_DU):
    """
    Xarray wrapper for DustEmissionK14.
    Returns a dataset or multiple dataarrays? apply_ufunc can return tuple.
    DustEmissionK14 returns: emission_slab, u, u_t, u_ts, R, H_w, f_erod
    """
    out = xr.apply_ufunc(
        DustEmissionK14,
        t_soil, w_top, rho_air, z0, z, u_z, v_z, ustar,
        f_land, f_snow, f_src, f_sand, f_silt, f_clay,
        texture, vegetation, gvf,
        f_w, f_c, uts_gamma, UNDEF, GRAV, VON_KARMAN,
        opt_clay, Ch_DU,
        input_core_dims=[
            ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'],
            ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'],
            ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'],
            [], [], [], [], [], [],
            [], []
        ],
        output_core_dims=[
            ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon']
        ],
        dask='parallelized',
        output_dtypes=[float, float, float, float, float, float, float]
    )

    # Pack into a Dataset? Or return tuple.
    # Users might prefer a tuple of DataArrays or a Dataset.
    # Let's return a tuple for now matching the python function.
    return out
