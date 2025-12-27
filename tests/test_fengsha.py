import unittest
import math
import numpy as np
import pyfengsha
from numba import jit
from pyfengsha.fengsha import (
    _calculate_drag_partition,
    _darmenova_drag_partition_vectorized,
    _leung_drag_partition_vectorized,
)


# The original, loop-based implementation for comparison
@jit(nopython=True)
def dust_emission_fengsha_original(
    fraclake: np.ndarray,
    fracsnow: np.ndarray,
    oro: np.ndarray,
    slc: np.ndarray,
    clay: np.ndarray,
    sand: np.ndarray,
    ssm: np.ndarray,
    rdrag: np.ndarray,
    airdens: np.ndarray,
    ustar: np.ndarray,
    vegfrac: np.ndarray,
    lai: np.ndarray,
    uthrs: np.ndarray,
    alpha: float,
    gamma: float,
    kvhmax: float,
    grav: float,
    distribution: np.ndarray,
    drylimit_factor: float,
    moist_correct: float,
    drag_opt: int,
) -> np.ndarray:
    ni, nj = fraclake.shape
    nbins = len(distribution)
    emissions = np.zeros((ni, nj, nbins))
    alpha_grav = alpha / max(grav, 1.0e-10)

    # Constants needed by the original function
    LAND_MASK_VALUE = 1.0
    VEG_THRESHOLD_FENGSHA = 0.4
    MAX_RDRAG_FENGSHA = 0.3
    SSM_THRESHOLD = 1.0e-02

    for j in range(nj):
        for i in range(ni):
            if (
                oro[i, j] != LAND_MASK_VALUE
                or (
                    drag_opt == 2
                    and (
                        vegfrac[i, j] < 0.0
                        or vegfrac[i, j] >= VEG_THRESHOLD_FENGSHA
                        or rdrag[i, j] > MAX_RDRAG_FENGSHA
                    )
                )
                or (
                    drag_opt == 3
                    and (vegfrac[i, j] < 0.0 or lai[i, j] >= VEG_THRESHOLD_FENGSHA)
                )
                or (drag_opt not in [2, 3] and rdrag[i, j] < 0.0)
                or ssm[i, j] < SSM_THRESHOLD
                or clay[i, j] < 0.0
                or sand[i, j] < 0.0
            ):
                continue

            fracland = max(0.0, 1.0 - fraclake[i, j]) * max(0.0, 1.0 - fracsnow[i, j])
            kvh = pyfengsha.mb95_vertical_flux_ratio(clay[i, j], kvhmax)
            total_emissions = (
                alpha_grav * fracland * (ssm[i, j] ** gamma) * airdens[i, j] * kvh
            )

            if drag_opt == 1:
                R = rdrag[i, j]
            elif drag_opt == 2:
                R = pyfengsha.darmenova_drag_partition(
                    rdrag[i, j], vegfrac[i, j], VEG_THRESHOLD_FENGSHA
                )
            elif drag_opt == 3:
                R = pyfengsha.leung_drag_partition(
                    rdrag[i, j], lai[i, j], vegfrac[i, j], VEG_THRESHOLD_FENGSHA
                )
            else:
                R = rdrag[i, j]

            rustar = R * ustar[i, j]
            smois = slc[i, j] * moist_correct
            h = pyfengsha.gocart_moisture_correction(
                smois, sand[i, j], clay[i, j], drylimit_factor
            )
            u_thresh = uthrs[i, j] * h

            u_sum = rustar + u_thresh
            q = max(0.0, rustar - u_thresh) * u_sum * u_sum

            for n in range(nbins):
                emissions[i, j, n] = distribution[n] * total_emissions * q

    return emissions


class TestFengsha(unittest.TestCase):
    def test_volumetric_to_gravimetric(self):
        vsoil = 0.2
        sandfrac = 0.5
        grav = pyfengsha.volumetric_to_gravimetric(vsoil, sandfrac)
        vsat = 0.489 - 0.00126 * (sandfrac * 100.0)
        expected_grav = vsoil * 1000.0 / (2650.0 * (1.0 - vsat))
        self.assertAlmostEqual(grav, expected_grav, places=4)

    def test_gocart_vol_to_grav(self):
        vsoil = 0.2
        sandfrac = 0.5
        grav = pyfengsha.gocart_vol_to_grav(vsoil, sandfrac)
        # vsat = 0.489 - 0.126 * sandfrac
        # expected_grav = vsoil * 1000.0 / (1700.0 * (1.0 - vsat))
        self.assertAlmostEqual(grav * 100, 20.496, places=2)

    def test_mb95_vertical_flux_ratio(self):
        clay = 0.1
        kvh = pyfengsha.mb95_vertical_flux_ratio(clay)
        expected = 10 ** (13.4 * 0.1 - 6.0)
        self.assertAlmostEqual(kvh, expected, places=5)

        clay = 0.3
        kvh = pyfengsha.mb95_vertical_flux_ratio(clay, max_ratio=0.0002)
        self.assertEqual(kvh, 0.0002)

    def test_kok_aerosol_distribution(self):
        radius = np.array([1.0, 2.0, 3.0])
        rLow = radius - 0.5
        rUp = radius + 0.5
        dist = pyfengsha.kok_aerosol_distribution(radius, rLow, rUp)
        self.assertEqual(len(dist), 3)
        self.assertAlmostEqual(np.sum(dist), 1.0, places=5)

    def test_kok_aerosol_distribution_vectorization(self):
        """
        Verify that the vectorized `kok_aerosol_distribution` function produces
        the same output as the original, JIT-compiled loop version.
        """

        # --- Original, loop-based implementation for comparison ---
        @jit(nopython=True)
        def kok_aerosol_distribution_original(
            radius: np.ndarray, r_low: np.ndarray, r_up: np.ndarray
        ) -> np.ndarray:
            median_mass_diameter = 3.4
            geom_std_dev = 3.0
            crack_prop_len = 12.0
            factor = 1.0 / (np.sqrt(2.0) * np.log(geom_std_dev))
            num_bins = len(radius)
            distribution = np.zeros(num_bins, dtype=np.float64)
            total_volume = 0.0
            for n in range(num_bins):
                diameter = 2.0 * radius[n]
                dlam = diameter / crack_prop_len
                # Numba requires `math.erf` for scalar op
                dist_val = (
                    diameter
                    * (1.0 + math.erf(factor * np.log(diameter / median_mass_diameter)))
                    * np.exp(-(dlam**3))
                    * np.log(r_up[n] / r_low[n])
                )
                distribution[n] = dist_val
                total_volume += dist_val
            if total_volume > 0:
                return distribution / total_volume
            return distribution

        # --- Test Data Setup ---
        np.random.seed(0)
        nbins = 10
        radius = np.random.uniform(0.1, 5.0, nbins)
        r_low = radius - 0.05
        r_up = radius + 0.05

        # --- Run both versions and compare ---
        dist_original = kok_aerosol_distribution_original(radius, r_low, r_up)
        dist_vectorized = pyfengsha.kok_aerosol_distribution(radius, r_low, r_up)

        # --- Verification ---
        self.assertEqual(dist_original.shape, dist_vectorized.shape)
        np.testing.assert_allclose(
            dist_original,
            dist_vectorized,
            rtol=1e-12,
            atol=1e-12,
            err_msg="Mismatch between original and vectorized Kok distribution.",
        )

    def test_dust_emission_fengsha(self):
        ni, nj, nbins = 2, 2, 3
        fraclake, fracsnow = np.zeros((ni, nj)), np.zeros((ni, nj))
        oro, slc = np.ones((ni, nj)), np.full((ni, nj), 0.0)
        clay, sand = np.full((ni, nj), 0.1), np.full((ni, nj), 0.8)
        ssm, rdrag = np.full((ni, nj), 1.0), np.full((ni, nj), 1.0)
        airdens, ustar = np.full((ni, nj), 1.2), np.full((ni, nj), 0.5)
        vegfrac, lai = np.zeros((ni, nj)), np.zeros((ni, nj))
        uthrs = np.full((ni, nj), 0.2)
        distribution = np.ones(nbins) / nbins

        emissions = pyfengsha.dust_emission_fengsha(
            fraclake,
            fracsnow,
            oro,
            slc,
            clay,
            sand,
            ssm,
            rdrag,
            airdens,
            ustar,
            vegfrac,
            lai,
            uthrs,
            1.0,
            1.0,
            1.0,
            9.81,
            distribution,
            1.0,
            1.0,
            1,
        )

        self.assertEqual(emissions.shape, (ni, nj, nbins))
        self.assertTrue(np.all(emissions >= 0))
        self.assertTrue(emissions[0, 0, 0] > 0)

        # Regression test with a known-good value
        expected_emission = 1.31131889e-07
        self.assertAlmostEqual(emissions[0, 0, 0], expected_emission, places=12)

    def test_dust_emission_fengsha_vectorization(self):
        """
        Verify that the vectorized `dust_emission_fengsha` function produces
        the same output as the original, JIT-compiled loop version.
        """
        # --- Test Data Setup ---
        ni, nj, nbins = 10, 15, 4
        np.random.seed(0)  # for reproducibility

        # Generate realistic random data for all input arrays
        fraclake = np.random.uniform(0, 0.1, (ni, nj))
        fracsnow = np.random.uniform(0, 0.2, (ni, nj))
        oro = np.random.choice([0.0, 1.0], (ni, nj), p=[0.1, 0.9])
        slc = np.random.uniform(0.01, 0.5, (ni, nj))
        clay = np.random.uniform(0.05, 0.4, (ni, nj))
        sand = np.random.uniform(0.1, 0.8, (ni, nj))
        ssm = np.random.uniform(0.1, 1.0, (ni, nj))
        rdrag = np.random.uniform(0.1, 0.3, (ni, nj))
        airdens = np.random.uniform(1.1, 1.3, (ni, nj))
        ustar = np.random.uniform(0.01, 0.8, (ni, nj))
        vegfrac = np.random.uniform(0, 0.5, (ni, nj))
        lai = np.random.uniform(0, 5, (ni, nj))
        uthrs = np.random.uniform(0.1, 0.3, (ni, nj))
        distribution = np.random.rand(nbins)
        distribution /= np.sum(distribution)  # Normalize

        # Scalar parameters
        params = {
            "alpha": 1.0e-5,
            "gamma": 1.0,
            "kvhmax": 0.0002,
            "grav": 9.81,
            "drylimit_factor": 1.0,
            "moist_correct": 1.0,
        }

        # --- Run for each drag option and compare ---
        for drag_opt in [1, 2, 3]:
            with self.subTest(drag_opt=drag_opt):
                # Run the original, JIT-compiled loop-based version
                emissions_original = dust_emission_fengsha_original(
                    fraclake,
                    fracsnow,
                    oro,
                    slc,
                    clay,
                    sand,
                    ssm,
                    rdrag,
                    airdens,
                    ustar,
                    vegfrac,
                    lai,
                    uthrs,
                    **params,
                    distribution=distribution,
                    drag_opt=drag_opt,
                )

                # Run the new vectorized version
                emissions_vectorized = pyfengsha.dust_emission_fengsha(
                    fraclake,
                    fracsnow,
                    oro,
                    slc,
                    clay,
                    sand,
                    ssm,
                    rdrag,
                    airdens,
                    ustar,
                    vegfrac,
                    lai,
                    uthrs,
                    **params,
                    distribution=distribution,
                    drag_opt=drag_opt,
                )

                # --- Verification ---
                # Check that the shapes are identical
                self.assertEqual(emissions_original.shape, emissions_vectorized.shape)

                # Check that the outputs are numerically very close.
                # A small tolerance (rtol) is used to account for potential
                # floating-point differences between the pure Python math in the
                # Numba JIT version and the vectorized NumPy operations.
                np.testing.assert_allclose(
                    emissions_original,
                    emissions_vectorized,
                    rtol=1e-12,
                    atol=1e-12,
                    err_msg=f"Mismatch for drag_opt={drag_opt}",
                )

    def test_dust_emission_gocart2g(self):
        ni, nj, nbins = 2, 2, 1
        radius = np.array([1.0e-6])
        fraclake, gwettop = np.zeros((ni, nj)), np.full((ni, nj), 0.1)
        oro, u10m, v10m = np.ones((ni, nj)), np.full((ni, nj), 5.0), np.zeros((ni, nj))
        du_src = np.ones((ni, nj))

        emissions = pyfengsha.dust_emission_gocart2g(
            radius, fraclake, gwettop, oro, u10m, v10m, 1.0e-9, du_src, 9.81
        )

        self.assertEqual(emissions.shape, (ni, nj, nbins))
        self.assertTrue(emissions[0, 0, 0] > 0)

    def test_dust_emission_gocart2g_vectorization(self):
        """
        Verify that the vectorized `dust_emission_gocart2g` function produces
        the same output as the original, JIT-compiled loop version.
        """

        # --- Original, loop-based implementation for comparison ---
        @jit(nopython=True)
        def dust_emission_gocart2g_original(
            radius: np.ndarray,
            fraclake: np.ndarray,
            gwettop: np.ndarray,
            oro: np.ndarray,
            u10m: np.ndarray,
            v10m: np.ndarray,
            Ch_DU: float,
            du_src: np.ndarray,
            grav: float,
        ) -> np.ndarray:
            air_dens = 1.25
            soil_density = 2.65 * 1000.0
            nbins = len(radius)
            ni, nj = u10m.shape
            emissions = np.zeros((ni, nj, nbins))
            LAND_MASK_VALUE = 1.0

            for n in range(nbins):
                diameter = 2.0 * radius[n]
                u_thresh0 = (
                    0.13
                    * np.sqrt(soil_density * grav * diameter / air_dens)
                    * np.sqrt(1.0 + 6.0e-7 / (soil_density * grav * diameter**2.5))
                    / np.sqrt(
                        1.928 * (1331.0 * (100.0 * diameter) ** 1.56 + 0.38) ** 0.092
                        - 1.0
                    )
                )

                for j in range(nj):
                    for i in range(ni):
                        if oro[i, j] != LAND_MASK_VALUE:
                            continue

                        w10m = np.sqrt(u10m[i, j] ** 2 + v10m[i, j] ** 2)
                        if gwettop[i, j] < 0.5:
                            u_thresh = max(
                                0.0,
                                u_thresh0
                                * (1.2 + 0.2 * np.log10(max(1.0e-3, gwettop[i, j]))),
                            )
                            if w10m > u_thresh:
                                emissions[i, j, n] = (
                                    (1.0 - fraclake[i, j]) * w10m**2 * (w10m - u_thresh)
                                )

            for j in range(nj):
                for i in range(ni):
                    emissions[i, j, :] *= Ch_DU * du_src[i, j]

            return emissions

        # --- Test Data Setup ---
        ni, nj, nbins = 10, 15, 4
        np.random.seed(0)

        radius = np.random.uniform(1e-7, 1e-5, nbins)
        fraclake = np.random.uniform(0, 0.1, (ni, nj))
        gwettop = np.random.uniform(0, 0.6, (ni, nj))
        oro = np.random.choice([0.0, 1.0], (ni, nj), p=[0.1, 0.9])
        u10m = np.random.uniform(-10, 10, (ni, nj))
        v10m = np.random.uniform(-10, 10, (ni, nj))
        du_src = np.random.uniform(0.5, 1.0, (ni, nj))
        Ch_DU = 1.0e-9
        grav = 9.81

        # --- Run both versions and compare ---
        emissions_original = dust_emission_gocart2g_original(
            radius, fraclake, gwettop, oro, u10m, v10m, Ch_DU, du_src, grav
        )
        emissions_vectorized = pyfengsha.dust_emission_gocart2g(
            radius, fraclake, gwettop, oro, u10m, v10m, Ch_DU, du_src, grav
        )

        # --- Verification ---
        self.assertEqual(emissions_original.shape, emissions_vectorized.shape)
        np.testing.assert_allclose(
            emissions_original,
            emissions_vectorized,
            rtol=1e-12,
            atol=1e-12,
            err_msg="Mismatch between original and vectorized GOCART2G implementations.",
        )


class TestFengshaHelpers(unittest.TestCase):
    def test_fecan_dry_limit(self):
        self.assertAlmostEqual(
            pyfengsha.fecan_dry_limit(0.0), 14.0 * 1e-4**2 + 17.0 * 1e-4
        )
        self.assertAlmostEqual(
            pyfengsha.fecan_dry_limit(0.2), 14.0 * 0.2**2 + 17.0 * 0.2
        )

    def test_fecan_moisture_correction(self):
        sand, clay, vol_soil_moisture = 0.9, 0.01, 0.3
        gravsm = pyfengsha.volumetric_to_gravimetric(vol_soil_moisture, sand)
        drylimit = pyfengsha.fecan_dry_limit(clay)
        self.assertTrue(gravsm > drylimit)
        H = pyfengsha.fecan_moisture_correction(vol_soil_moisture, sand, clay)
        expected_H = np.sqrt(1.0 + 1.21 * (gravsm - drylimit) ** 0.68)
        self.assertAlmostEqual(H, expected_H)

        sand, clay, vol_soil_moisture = 0.5, 0.2, 0.01
        gravsm = pyfengsha.volumetric_to_gravimetric(vol_soil_moisture, sand)
        drylimit = pyfengsha.fecan_dry_limit(clay)
        self.assertTrue(gravsm < drylimit)
        self.assertEqual(
            pyfengsha.fecan_moisture_correction(vol_soil_moisture, sand, clay), 1.0
        )

    def test_shao_1996_soil_moisture(self):
        self.assertAlmostEqual(
            pyfengsha.shao_1996_soil_moisture(0.1), np.exp(22.7 * 0.1)
        )

    def test_shao_2004_soil_moisture(self):
        self.assertAlmostEqual(
            pyfengsha.shao_2004_soil_moisture(0.02), np.exp(22.7 * 0.02)
        )
        self.assertAlmostEqual(
            pyfengsha.shao_2004_soil_moisture(0.04), np.exp(95.3 * 0.04 - 2.029)
        )

    def test_modified_threshold_velocity(self):
        self.assertAlmostEqual(
            pyfengsha.modified_threshold_velocity(0.2, 1.5, 0.8), 0.375
        )

    def test_horizontal_saltation_flux(self):
        self.assertAlmostEqual(
            pyfengsha.horizontal_saltation_flux(0.5, 0.2), 0.5 * (0.5**2 - 0.2**2)
        )
        self.assertEqual(pyfengsha.horizontal_saltation_flux(0.2, 0.5), 0.0)

    def test_mackinnon_drag_partition(self):
        z0, z0s = 0.001, 1.0e-04
        expected = 1.0 - np.log(z0 / z0s) / np.log(0.7 * (12255.0 / z0s) ** 0.8)
        self.assertAlmostEqual(pyfengsha.mackinnon_drag_partition(z0), expected)

    def test_mb95_drag_partition(self):
        z0, z0s = 0.001, 1.0e-04
        expected = 1.0 - np.log(z0 / z0s) / np.log(0.7 * (10.0 / z0s) ** 0.8)
        self.assertAlmostEqual(pyfengsha.mb95_drag_partition(z0), expected)

    def test_fengsha_albedo(self):
        self.assertEqual(
            pyfengsha.fengsha_albedo(1.2, 0.1, 0.0, 1, 0.5, 0.2, 0.5, 0.8, 0.2), 0.0
        )
        self.assertEqual(
            pyfengsha.fengsha_albedo(1.2, 0.1, 1.0, 0, 0.5, 0.2, 0.5, 0.8, 0.2), 0.0
        )

        rho, smois, ssm, xland, ust, clay, sand, rdrag, u_ts0 = (
            1.2,
            0.1,
            1.0,
            1,
            0.5,
            0.2,
            0.5,
            0.8,
            0.2,
        )
        H = pyfengsha.fecan_moisture_correction(smois, sand, clay)
        kvh = pyfengsha.mb95_vertical_flux_ratio(clay)
        u_ts = pyfengsha.modified_threshold_velocity(u_ts0, H, rdrag)
        ustar_albedo = ust * rdrag
        Q = pyfengsha.horizontal_saltation_flux(ustar_albedo, u_ts)
        expected = ssm * rho / (9.81 * 100.0) * kvh * Q
        self.assertAlmostEqual(
            pyfengsha.fengsha_albedo(
                rho, smois, ssm, xland, ust, clay, sand, rdrag, u_ts0
            ),
            expected,
        )

    def test_darmenova_drag_partition(self):
        self.assertAlmostEqual(
            pyfengsha.darmenova_drag_partition(0.1, 0.5, 0.4), 1.0e-3
        )
        feff = pyfengsha.darmenova_drag_partition(0.1, 0.2, 0.4)
        self.assertTrue(1.0e-5 < feff < 1.0)

    def test_leung_drag_partition(self):
        self.assertAlmostEqual(
            pyfengsha.leung_drag_partition(0.1, 0.5, 0.5, 0.4), 1.0e-5, places=4
        )
        feff = pyfengsha.leung_drag_partition(0.1, 0.2, 0.8, 0.4)
        self.assertTrue(1.0e-5 < feff < 1.0)


# Find the private function for testing


class TestDragPartitionHelpers(unittest.TestCase):
    def setUp(self):
        """Set up common test data."""
        self.n_pts = 20
        np.random.seed(0)
        self.rdrag = np.random.uniform(0.1, 0.3, self.n_pts)
        self.vegfrac = np.random.uniform(0, 0.5, self.n_pts)
        self.lai = np.random.uniform(0, 5, self.n_pts)
        self.VEG_THRESHOLD_FENGSHA = 0.4

    def test_darmenova_vectorized(self):
        """
        Test the vectorized Darmenova implementation against the original
        scalar function.
        """
        expected = np.array(
            [
                pyfengsha.darmenova_drag_partition(r, v, self.VEG_THRESHOLD_FENGSHA)
                for r, v in zip(self.rdrag, self.vegfrac)
            ]
        )
        actual = _darmenova_drag_partition_vectorized(self.rdrag, self.vegfrac)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_leung_vectorized(self):
        """
        Test the vectorized Leung implementation against the original scalar
        function.
        """
        expected = np.array(
            [
                pyfengsha.leung_drag_partition(
                    r, lai_val, v, self.VEG_THRESHOLD_FENGSHA
                )
                for r, lai_val, v in zip(self.rdrag, self.lai, self.vegfrac)
            ]
        )
        actual = _leung_drag_partition_vectorized(self.rdrag, self.vegfrac, self.lai)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


class TestCalculateDragPartition(unittest.TestCase):
    def setUp(self):
        """Set up common test data."""
        self.n_pts = 20
        np.random.seed(0)
        self.rdrag = np.random.uniform(0.1, 0.3, self.n_pts)
        self.vegfrac = np.random.uniform(0, 0.5, self.n_pts)
        self.lai = np.random.uniform(0, 5, self.n_pts)

    def test_drag_opt_1(self):
        """Test drag_opt=1 (default) returns rdrag unchanged."""
        result = _calculate_drag_partition(self.rdrag, self.vegfrac, self.lai, 1)
        np.testing.assert_array_equal(result, self.rdrag)

    def test_drag_opt_2_darmenova(self):
        """
        Test drag_opt=2, comparing the vectorized implementation against the
        original scalar darmenova_drag_partition function.
        """
        # Calculate expected values using the original scalar function
        VEG_THRESHOLD_FENGSHA = 0.4
        expected = np.array(
            [
                pyfengsha.darmenova_drag_partition(r, v, VEG_THRESHOLD_FENGSHA)
                for r, v in zip(self.rdrag, self.vegfrac)
            ]
        )

        # Calculate actual values using the vectorized helper
        actual = _calculate_drag_partition(self.rdrag, self.vegfrac, self.lai, 2)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_drag_opt_3_leung(self):
        """
        Test drag_opt=3, comparing the vectorized implementation against the
        original scalar leung_drag_partition function.
        """
        # Calculate expected values using the original scalar function
        VEG_THRESHOLD_FENGSHA = 0.4  # Original code uses this as LAI threshold
        expected = np.array(
            [
                pyfengsha.leung_drag_partition(r, lai_val, v, VEG_THRESHOLD_FENGSHA)
                for r, lai_val, v in zip(self.rdrag, self.lai, self.vegfrac)
            ]
        )

        # Calculate actual values using the vectorized helper
        actual = _calculate_drag_partition(self.rdrag, self.vegfrac, self.lai, 3)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
