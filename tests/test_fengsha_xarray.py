import unittest
import numpy as np
import pyfengsha
import xarray as xr

class TestFengshaXarray(unittest.TestCase):
    def test_DustEmissionFENGSHA_xr(self):
        ni, nj, nbins = 2, 2, 3
        coords = {'lat': np.arange(ni), 'lon': np.arange(nj)}
        coords_bin = {'bin': np.arange(nbins)}

        def make_da(val, shape=(ni, nj), dims=('lat', 'lon')):
            return xr.DataArray(np.full(shape, val), coords=coords, dims=dims)

        fraclake, fracsnow, oro = make_da(0.0), make_da(0.0), make_da(1.0)
        slc, clay, sand = make_da(0.0), make_da(0.1), make_da(0.8)
        ssm, rdrag, airdens = make_da(1.0), make_da(1.0), make_da(1.2)
        ustar, vegfrac, lai, uthrs = make_da(0.5), make_da(0.0), make_da(0.0), make_da(0.2)

        distribution = xr.DataArray(np.ones(nbins) / nbins, coords=coords_bin, dims='bin')

        emissions = pyfengsha.DustEmissionFENGSHA_xr(
            fraclake, fracsnow, oro, slc, clay, sand,
            ssm, rdrag, airdens, ustar, vegfrac, lai, uthrs,
            1.0, 1.0, 1.0, 9.81, distribution,
            1.0, 1.0, 1
        )

        self.assertIsInstance(emissions, xr.DataArray)
        self.assertEqual(emissions.shape, (ni, nj, nbins))
        self.assertEqual(emissions.dims, ('lat', 'lon', 'bin'))
        self.assertTrue((emissions > 0).all())

        nt = 2
        ustar_t = ustar.expand_dims(time=np.arange(nt))

        emissions_t = pyfengsha.DustEmissionFENGSHA_xr(
            fraclake, fracsnow, oro, slc, clay, sand,
            ssm, rdrag, airdens, ustar_t, vegfrac, lai, uthrs,
            1.0, 1.0, 1.0, 9.81, distribution,
            1.0, 1.0, 1
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
