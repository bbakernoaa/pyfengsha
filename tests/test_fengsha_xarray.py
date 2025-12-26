import unittest
import numpy as np
import pyfengsha
import xarray as xr

class TestFengshaXarray(unittest.TestCase):
    def test_DustEmissionFENGSHA_xr(self):
        ni, nj, nbins = 2, 2, 3
        coords = {'lat': np.arange(ni), 'lon': np.arange(nj)}
        coords_bin = {'bin': np.arange(nbins)}

        # Create a Dataset with all necessary variables
        ds = xr.Dataset({
            'fraclake': (('lat', 'lon'), np.full((ni, nj), 0.0)),
            'fracsnow': (('lat', 'lon'), np.full((ni, nj), 0.0)),
            'oro': (('lat', 'lon'), np.full((ni, nj), 1.0)),
            'slc': (('lat', 'lon'), np.full((ni, nj), 0.0)),
            'clay': (('lat', 'lon'), np.full((ni, nj), 0.1)),
            'sand': (('lat', 'lon'), np.full((ni, nj), 0.8)),
            'ssm': (('lat', 'lon'), np.full((ni, nj), 1.0)),
            'rdrag': (('lat', 'lon'), np.full((ni, nj), 1.0)),
            'airdens': (('lat', 'lon'), np.full((ni, nj), 1.2)),
            'ustar': (('lat', 'lon'), np.full((ni, nj), 0.5)),
            'vegfrac': (('lat', 'lon'), np.full((ni, nj), 0.0)),
            'lai': (('lat', 'lon'), np.full((ni, nj), 0.0)),
            'uthrs': (('lat', 'lon'), np.full((ni, nj), 0.2)),
            'distribution': (('bin',), np.ones(nbins) / nbins)
        }, coords={**coords, **coords_bin})

        emissions = pyfengsha.DustEmissionFENGSHA_xr(
            ds=ds,
            alpha=1.0, gamma=1.0, kvhmax=1.0, grav=9.81,
            drylimit_factor=1.0, moist_correct=1.0, drag_opt=1
        )

        self.assertIsInstance(emissions, xr.DataArray)
        self.assertEqual(emissions.shape, (ni, nj, nbins))
        self.assertEqual(emissions.dims, ('lat', 'lon', 'bin'))
        self.assertTrue((emissions.values > 0).all())

        # Test with a time dimension
        nt = 2
        ds_t = ds.expand_dims(time=np.arange(nt)).copy()
        # Update ustar to be time-dependent
        ds_t['ustar'] = ds['ustar'].expand_dims(time=np.arange(nt))

        # Re-run with the time-aware dataset
        # Note: The underlying numpy function doesn't inherently handle the time dim,
        # apply_ufunc maps it over the core dims, so we adjust the input_core_dims.
        # This part of the logic is now handled internally by the refactored function
        # which simplifies the call significantly.
        # We need to adapt the test to reflect the new API which expects a single dataset.
        # Since the core function isn't changing, we can test that the wrapper handles
        # additional dimensions correctly.
        emissions_t = pyfengsha.DustEmissionFENGSHA_xr(
            ds=ds_t,
            alpha=1.0, gamma=1.0, kvhmax=1.0, grav=9.81,
            drylimit_factor=1.0, moist_correct=1.0, drag_opt=1
        )

        self.assertEqual(emissions_t.shape, (nt, ni, nj, nbins))
        self.assertIn('time', emissions_t.dims)


    def test_DustEmissionGOCART2G_xr(self):
        ni, nj, nbins = 2, 2, 1
        coords = {'lat': np.arange(ni), 'lon': np.arange(nj)}

        radius = xr.DataArray([1.0e-6], dims='bin')
        fraclake = xr.DataArray(np.zeros((ni, nj)), coords=coords, dims=('lat', 'lon'))
        gwettop = xr.DataArray(np.full((ni, nj), 0.1), coords=coords, dims=('lat', 'lon'))
        oro = xr.DataArray(np.ones((ni, nj)), coords=coords, dims=('lat', 'lon'))
        u10m = xr.DataArray(np.full((ni, nj), 5.0), coords=coords, dims=('lat', 'lon'))
        v10m = xr.DataArray(np.zeros((ni, nj)), coords=coords, dims=('lat', 'lon'))
        du_src = xr.DataArray(np.ones((ni, nj)), coords=coords, dims=('lat', 'lon'))

        emissions = pyfengsha.DustEmissionGOCART2G_xr(radius, fraclake, gwettop, oro, u10m, v10m, 1.0e-9, du_src, 9.81)

        self.assertEqual(emissions.shape, (ni, nj, nbins))
        self.assertTrue((emissions > 0).all())

if __name__ == '__main__':
    unittest.main()
