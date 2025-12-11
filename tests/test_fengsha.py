import unittest
import numpy as np
import pyfengsha

class TestFengsha(unittest.TestCase):
    def test_volumetric_to_gravimetric(self):
        vsoil = 0.2
        sandfrac = 0.5
        grav = pyfengsha.volumetric_to_gravimetric(vsoil, sandfrac)
        # Expected calculation:
        # vsat = 0.489 - 0.00126 * 50 = 0.489 - 0.063 = 0.426
        # grav = 0.2 * 1000 / (2650 * (1 - 0.426)) = 200 / (2650 * 0.574) = 200 / 1521.1 = 0.13148
        self.assertAlmostEqual(grav, 0.13148, places=4)

    def test_soilMoistureConvertVol2Grav(self):
        vsoil = 0.2
        sandfrac = 0.5
        grav = pyfengsha.soilMoistureConvertVol2Grav(vsoil, sandfrac)
        # GOCART logic:
        # vsat = 0.489 - 0.126 * 0.5 = 0.489 - 0.063 = 0.426
        # grav = 100 * 0.2 * 1000 / (1700 * (1 - 0.426)) = 20000 / (1700 * 0.574) = 20000 / 975.8 = 20.496
        self.assertAlmostEqual(grav, 20.496, places=2)

    def test_DustFluxV2HRatioMB95(self):
        clay = 0.1
        max_ratio = 1.0
        kvh = pyfengsha.DustFluxV2HRatioMB95(clay, max_ratio)
        expected = 10**(13.4 * 0.1 - 6.0) # 10^(-4.66)
        self.assertAlmostEqual(kvh, expected, places=5)

        clay = 0.3
        kvh = pyfengsha.DustFluxV2HRatioMB95(clay, max_ratio)
        self.assertEqual(kvh, max_ratio)

    def test_DustAerosolDistributionKok(self):
        radius = np.array([1.0, 2.0, 3.0])
        rLow = radius - 0.5
        rUp = radius + 0.5
        dist = pyfengsha.DustAerosolDistributionKok(radius, rLow, rUp)
        self.assertEqual(len(dist), 3)
        self.assertAlmostEqual(np.sum(dist), 1.0, places=5)

    def test_DustEmissionFENGSHA(self):
        ni, nj = 2, 2
        nbins = 3
        fraclake = np.zeros((ni, nj))
        fracsnow = np.zeros((ni, nj))
        oro = np.ones((ni, nj)) # LAND
        slc = np.full((ni, nj), 0.0) # Dry soil
        clay = np.full((ni, nj), 0.1)
        sand = np.full((ni, nj), 0.8)
        silt = np.full((ni, nj), 0.1)
        ssm = np.full((ni, nj), 1.0)
        rdrag = np.full((ni, nj), 1.0) # High drag partition ratio
        airdens = np.full((ni, nj), 1.2)
        ustar = np.full((ni, nj), 0.5)
        vegfrac = np.zeros((ni, nj))
        lai = np.zeros((ni, nj))
        uthrs = np.full((ni, nj), 0.2)
        alpha = 1.0
        gamma = 1.0
        kvhmax = 1.0
        grav = 9.81
        rhop = np.ones(nbins) * 2650.0
        distribution = np.ones(nbins) / nbins
        drylimit_factor = 1.0
        moist_correct = 1.0
        drag_opt = 1

        emissions = pyfengsha.DustEmissionFENGSHA(fraclake, fracsnow, oro, slc, clay, sand, silt,
                                                  ssm, rdrag, airdens, ustar, vegfrac, lai, uthrs,
                                                  alpha, gamma, kvhmax, grav, rhop, distribution,
                                                  drylimit_factor, moist_correct, drag_opt)

        self.assertEqual(emissions.shape, (ni, nj, nbins))
        self.assertTrue(np.all(emissions >= 0))
        # Check non-zero emission where conditions permit
        self.assertTrue(emissions[0,0,0] > 0)

    def test_DustEmissionGOCART2G(self):
        ni, nj = 2, 2
        nbins = 1
        radius = np.array([1.0e-6])
        fraclake = np.zeros((ni, nj))
        gwettop = np.full((ni, nj), 0.1)
        oro = np.ones((ni, nj))
        u10m = np.full((ni, nj), 5.0)
        v10m = np.full((ni, nj), 0.0)
        Ch_DU = 1.0e-9
        du_src = np.ones((ni, nj))
        grav = 9.81

        emissions = pyfengsha.DustEmissionGOCART2G(radius, fraclake, gwettop, oro, u10m, v10m, Ch_DU, du_src, grav)

        self.assertEqual(emissions.shape, (ni, nj, nbins))
        self.assertTrue(emissions[0,0,0] > 0)

if __name__ == '__main__':
    unittest.main()
