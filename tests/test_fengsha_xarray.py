import unittest
import numpy as np
import pyfengsha
import xarray as xr


class TestFengshaXarray(unittest.TestCase):
    def test_DustEmissionFENGSHA_xr(self):
        ni, nj, nbins = 2, 2, 3
        coords = {"lat": np.arange(ni), "lon": np.arange(nj)}
        coords_bin = {"bin": np.arange(nbins)}

        # Create a Dataset with all necessary variables
        ds = xr.Dataset(
            {
                "fraclake": (("lat", "lon"), np.full((ni, nj), 0.0)),
                "fracsnow": (("lat", "lon"), np.full((ni, nj), 0.0)),
                "oro": (("lat", "lon"), np.full((ni, nj), 1.0)),
                "slc": (("lat", "lon"), np.full((ni, nj), 0.0)),
                "clay": (("lat", "lon"), np.full((ni, nj), 0.1)),
                "sand": (("lat", "lon"), np.full((ni, nj), 0.8)),
                "ssm": (("lat", "lon"), np.full((ni, nj), 1.0)),
                "rdrag": (("lat", "lon"), np.full((ni, nj), 1.0)),
                "airdens": (("lat", "lon"), np.full((ni, nj), 1.2)),
                "ustar": (("lat", "lon"), np.full((ni, nj), 0.5)),
                "vegfrac": (("lat", "lon"), np.full((ni, nj), 0.0)),
                "lai": (("lat", "lon"), np.full((ni, nj), 0.0)),
                "uthrs": (("lat", "lon"), np.full((ni, nj), 0.2)),
                "distribution": (("bin",), np.ones(nbins) / nbins),
            },
            coords={**coords, **coords_bin},
        )

        emissions = pyfengsha.DustEmissionFENGSHA_xr(
            ds=ds,
            alpha=1.0,
            gamma=1.0,
            kvhmax=1.0,
            grav=9.81,
            drylimit_factor=1.0,
            moist_correct=1.0,
            drag_opt=1,
        )

        self.assertIsInstance(emissions, xr.DataArray)
        self.assertEqual(emissions.shape, (ni, nj, nbins))
        self.assertEqual(emissions.dims, ("lat", "lon", "bin"))
        self.assertTrue((emissions.values > 0).all())

        # Test with a time dimension
        nt = 2
        ds_t = ds.expand_dims(time=np.arange(nt)).copy(deep=True)
        # Update ustar to be time-dependent, but keep distribution time-invariant
        ds_t["ustar"] = ds["ustar"].expand_dims(time=np.arange(nt))
        ds_t["distribution"] = ds["distribution"]

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
            alpha=1.0,
            gamma=1.0,
            kvhmax=1.0,
            grav=9.81,
            drylimit_factor=1.0,
            moist_correct=1.0,
            drag_opt=1,
        )

        self.assertEqual(emissions_t.shape, (nt, ni, nj, nbins))
        self.assertIn("time", emissions_t.dims)

    def test_DustEmissionGOCART2G_xr(self):
        ni, nj, nbins = 2, 2, 1
        coords = {"lat": np.arange(ni), "lon": np.arange(nj)}
        coords_bin = {"bin": np.arange(nbins)}

        # Create a Dataset with all necessary variables
        ds = xr.Dataset(
            {
                "radius": (("bin",), np.array([1.0e-6])),
                "fraclake": (("lat", "lon"), np.zeros((ni, nj))),
                "gwettop": (("lat", "lon"), np.full((ni, nj), 0.1)),
                "oro": (("lat", "lon"), np.ones((ni, nj))),
                "u10m": (("lat", "lon"), np.full((ni, nj), 5.0)),
                "v10m": (("lat", "lon"), np.zeros((ni, nj))),
                "du_src": (("lat", "lon"), np.ones((ni, nj))),
            },
            coords={**coords, **coords_bin},
        )

        emissions = pyfengsha.DustEmissionGOCART2G_xr(ds=ds, Ch_DU=1.0e-9, grav=9.81)

        self.assertEqual(emissions.shape, (ni, nj, nbins))
        self.assertTrue((emissions.values > 0).all())


if __name__ == "__main__":
    unittest.main()
