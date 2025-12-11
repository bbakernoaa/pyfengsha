import unittest
import numpy as np
import pyfengsha
try:
    import xarray as xr
    has_xarray = True
except ImportError:
    has_xarray = False

class TestFengshaXarray(unittest.TestCase):
    def setUp(self):
        if not has_xarray:
            self.skipTest("xarray not installed")

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

    def test_DustEmissionK14_xr(self):
        ni, nj = 2, 2
        coords = {'lat': np.arange(ni), 'lon': np.arange(nj)}

        def make_da(val):
            return xr.DataArray(np.full((ni, nj), val), coords=coords, dims=('lat', 'lon'))

        t_soil = make_da(300.0)
        w_top = make_da(0.1)
        rho_air = make_da(1.2)
        z0 = make_da(1e-4)
        z = make_da(10.0)
        u_z = make_da(5.0)
        v_z = make_da(0.0)
        ustar = make_da(0.5)
        f_land = make_da(1.0)
        f_snow = make_da(0.0)
        f_src = make_da(1.0)
        f_sand = make_da(0.8)
        f_silt = make_da(0.1)
        f_clay = make_da(0.1)
        texture = make_da(1.0) # Soil type 1
        vegetation = make_da(7.0) # Shrubland
        gvf = make_da(0.0)

        f_w = 1.0
        f_c = 1.0
        uts_gamma = 1.65e-4
        UNDEF = -999.0
        GRAV = 9.81
        VON_KARMAN = 0.4
        opt_clay = 1
        Ch_DU = 1.0e-9

        outputs = pyfengsha.DustEmissionK14_xr(
            t_soil, w_top, rho_air, z0, z, u_z, v_z, ustar,
            f_land, f_snow, f_src, f_sand, f_silt, f_clay,
            texture, vegetation, gvf,
            f_w, f_c, uts_gamma, UNDEF, GRAV, VON_KARMAN,
            opt_clay, Ch_DU
        )

        # Expect tuple of 7 outputs
        self.assertEqual(len(outputs), 7)
        emissions = outputs[0]
        self.assertEqual(emissions.shape, (ni, nj))
        # With provided parameters, we should get some emission
        # z0 is small (1e-4), z0s will be small.
        # ustar=0.5.

        # Debugging info if assertion fails
        msg = f"\nu={outputs[1].values}\nu_t={outputs[2].values}\nu_ts={outputs[3].values}\nR={outputs[4].values}\nH_w={outputs[5].values}\nf_erod={outputs[6].values}"
        self.assertTrue((emissions > 0).all(), msg)

if __name__ == '__main__':
    unittest.main()
