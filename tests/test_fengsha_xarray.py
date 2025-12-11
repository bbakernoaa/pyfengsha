import unittest
import numpy as np
import pyfengsha
import xarray as xr

class TestFengshaXarray(unittest.TestCase):
    def test_DustEmissionFENGSHA_xr(self):
        ni, nj = 2, 2
        nbins = 3
        coords = {'lat': np.arange(ni), 'lon': np.arange(nj)}
        coords_bin = {'bin': np.arange(nbins)}

        def make_da(val, shape=(ni, nj), dims=('lat', 'lon')):
            return xr.DataArray(np.full(shape, val), coords=coords, dims=dims)

        fraclake = make_da(0.0)
        fracsnow = make_da(0.0)
        oro = make_da(1.0)
        slc = make_da(0.0)
        clay = make_da(0.1)
        sand = make_da(0.8)
        silt = make_da(0.1)
        ssm = make_da(1.0)
        rdrag = make_da(1.0)
        airdens = make_da(1.2)
        ustar = make_da(0.5)
        vegfrac = make_da(0.0)
        lai = make_da(0.0)
        uthrs = make_da(0.2)

        # scalars or 1D arrays
        rhop = xr.DataArray(np.ones(nbins) * 2650.0, coords=coords_bin, dims='bin')
        distribution = xr.DataArray(np.ones(nbins) / nbins, coords=coords_bin, dims='bin')

        # scalars
        alpha = 1.0
        gamma = 1.0
        kvhmax = 1.0
        grav = 9.81
        drylimit_factor = 1.0
        moist_correct = 1.0
        drag_opt = 1

        emissions = pyfengsha.DustEmissionFENGSHA_xr(
            fraclake, fracsnow, oro, slc, clay, sand, silt,
            ssm, rdrag, airdens, ustar, vegfrac, lai, uthrs,
            alpha, gamma, kvhmax, grav, rhop, distribution,
            drylimit_factor, moist_correct, drag_opt
        )

        self.assertIsInstance(emissions, xr.DataArray)
        self.assertEqual(emissions.shape, (ni, nj, nbins))
        self.assertEqual(emissions.dims, ('lat', 'lon', 'bin'))
        self.assertTrue((emissions > 0).all())

        # Test broadcasting (e.g. time dimension)
        nt = 2
        ustar_t = ustar.expand_dims(time=np.arange(nt)) # (time, lat, lon)

        emissions_t = pyfengsha.DustEmissionFENGSHA_xr(
            fraclake, fracsnow, oro, slc, clay, sand, silt,
            ssm, rdrag, airdens, ustar_t, vegfrac, lai, uthrs,
            alpha, gamma, kvhmax, grav, rhop, distribution,
            drylimit_factor, moist_correct, drag_opt
        )

        self.assertEqual(emissions_t.shape, (nt, ni, nj, nbins))
        # Note: apply_ufunc usually puts broadcast dims first if not core.
        self.assertIn('time', emissions_t.dims)

    def test_DustEmissionGOCART2G_xr(self):
        ni, nj = 2, 2
        nbins = 1
        coords = {'lat': np.arange(ni), 'lon': np.arange(nj)}

        radius = xr.DataArray([1.0e-6], dims='bin')
        fraclake = xr.DataArray(np.zeros((ni, nj)), coords=coords, dims=('lat', 'lon'))
        gwettop = xr.DataArray(np.full((ni, nj), 0.1), coords=coords, dims=('lat', 'lon'))
        oro = xr.DataArray(np.ones((ni, nj)), coords=coords, dims=('lat', 'lon'))
        u10m = xr.DataArray(np.full((ni, nj), 5.0), coords=coords, dims=('lat', 'lon'))
        v10m = xr.DataArray(np.zeros((ni, nj)), coords=coords, dims=('lat', 'lon'))
        du_src = xr.DataArray(np.ones((ni, nj)), coords=coords, dims=('lat', 'lon'))

        Ch_DU = 1.0e-9
        grav = 9.81

        emissions = pyfengsha.DustEmissionGOCART2G_xr(radius, fraclake, gwettop, oro, u10m, v10m, Ch_DU, du_src, grav)

        self.assertEqual(emissions.shape, (ni, nj, nbins))
        self.assertTrue((emissions > 0).all())

if __name__ == '__main__':
    unittest.main()
