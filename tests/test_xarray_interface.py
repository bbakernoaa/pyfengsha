import unittest
import numpy as np
import xarray as xr
import dask.array as da
from pyfengsha.xarray_interface import (
    DustEmissionFENGSHA_xr,
    DustEmissionGOCART2G_xr,
)


class TestFENGSHAXarray(unittest.TestCase):
    def setUp(self):
        """Set up a sample xarray.Dataset for FENGSHA testing."""
        self.ni, self.nj, self.nbins = 10, 20, 3
        self.coords = {
            "lat": np.arange(self.ni),
            "lon": np.arange(self.nj),
            "bin": np.arange(self.nbins),
        }
        dist_data = np.array([0.1, 0.2, 0.7])
        self.ds = xr.Dataset(
            {
                "fraclake": (("lat", "lon"), da.zeros((self.ni, self.nj), chunks=(5, 10))),
                "fracsnow": (("lat", "lon"), da.zeros((self.ni, self.nj), chunks=(5, 10))),
                "oro": (("lat", "lon"), da.ones((self.ni, self.nj), chunks=(5, 10))),
                "slc": (("lat", "lon"), da.full((self.ni, self.nj), 0.2, chunks=(5, 10))),
                "clay": (("lat", "lon"), da.full((self.ni, self.nj), 0.1, chunks=(5, 10))),
                "sand": (("lat", "lon"), da.full((self.ni, self.nj), 0.8, chunks=(5, 10))),
                "ssm": (("lat", "lon"), da.full((self.ni, self.nj), 0.9, chunks=(5, 10))),
                "rdrag": (("lat", "lon"), da.full((self.ni, self.nj), 0.95, chunks=(5, 10))),
                "airdens": (("lat", "lon"), da.full((self.ni, self.nj), 1.2, chunks=(5, 10))),
                "ustar": (("lat", "lon"), da.full((self.ni, self.nj), 0.4, chunks=(5, 10))),
                "vegfrac": (("lat", "lon"), da.full((self.ni, self.nj), 0.1, chunks=(5, 10))),
                "lai": (("lat", "lon"), da.full((self.ni, self.nj), 0.2, chunks=(5, 10))),
                "uthrs": (("lat", "lon"), da.full((self.ni, self.nj), 0.25, chunks=(5, 10))),
                "distribution": (("bin",), da.from_array(dist_data, chunks=(self.nbins,))),
            },
            coords=self.coords,
        )
        self.ds.attrs["history"] = "2023-01-01T00:00:00Z: Initial dataset creation."

    def test_fengsha_dataset_input_and_provenance(self):
        """Test FENGSHA with a Dataset input and check for provenance."""
        emissions = DustEmissionFENGSHA_xr(
            ds=self.ds,
            alpha=1.0,
            gamma=1.0,
            kvhmax=2.0e-4,
            grav=9.81,
            drylimit_factor=1.0,
            moist_correct=1.0,
            drag_opt=1,
        )
        self.assertIsInstance(emissions, xr.DataArray)
        self.assertEqual(emissions.shape, (self.ni, self.nj, self.nbins))
        self.assertIn("history", emissions.attrs)
        self.assertIn("FENGSHA scheme", emissions.attrs["history"])
        self.assertTrue(hasattr(emissions.data, "dask"))
        computed_emissions = emissions.compute()
        self.assertFalse(np.isnan(computed_emissions).any())


class TestGOCART2GXarray(unittest.TestCase):
    def setUp(self):
        """Set up a sample xarray.Dataset for GOCART2G testing."""
        self.ni, self.nj, self.nbins = 10, 20, 3
        self.coords = {
            "lat": np.arange(self.ni),
            "lon": np.arange(self.nj),
            "bin": np.arange(self.nbins),
        }
        radius_data = np.array([0.1, 0.5, 1.0])
        self.ds = xr.Dataset(
            {
                "radius": (("bin",), da.from_array(radius_data, chunks=(self.nbins,))),
                "fraclake": (("lat", "lon"), da.zeros((self.ni, self.nj), chunks=(5, 10))),
                "gwettop": (("lat", "lon"), da.full((self.ni, self.nj), 0.1, chunks=(5, 10))),
                "oro": (("lat", "lon"), da.ones((self.ni, self.nj), chunks=(5, 10))),
                "u10m": (("lat", "lon"), da.full((self.ni, self.nj), 5.0, chunks=(5, 10))),
                "v10m": (("lat", "lon"), da.full((self.ni, self.nj), 2.0, chunks=(5, 10))),
                "du_src": (("lat", "lon"), da.ones((self.ni, self.nj), chunks=(5, 10))),
            },
            coords=self.coords,
        )
        self.ds.attrs["history"] = "2023-01-01T00:00:00Z: Initial GOCART2G dataset."

    def test_gocart2g_dataset_input_and_provenance(self):
        """Test the refactored GOCART2G function with a Dataset and check provenance."""
        emissions = DustEmissionGOCART2G_xr(
            ds=self.ds, Ch_DU=1.0e-5, grav=9.81
        )

        # 1. Verify Output Type and Shape
        self.assertIsInstance(emissions, xr.DataArray)
        self.assertEqual(emissions.shape, (self.ni, self.nj, self.nbins))
        self.assertEqual(emissions.dims, ("lat", "lon", "bin"))

        # 2. Verify Coordinates are Preserved
        dummy_array = xr.DataArray(
            da.zeros((self.ni, self.nj, self.nbins), chunks=(5, 10, self.nbins)),
            coords=self.ds.coords,
            dims=("lat", "lon", "bin"),
        )
        xr.testing.assert_allclose(
            emissions.coords.to_dataset(), dummy_array.coords.to_dataset()
        )

        # 3. Verify Provenance (History) Tracking
        self.assertIn("history", emissions.attrs)
        self.assertTrue(emissions.attrs["history"].startswith("20"))
        self.assertIn("GOCART2G scheme", emissions.attrs["history"])
        self.assertIn(self.ds.attrs["history"], emissions.attrs["history"])

        # 4. Verify Computation (Lazy Execution)
        self.assertTrue(hasattr(emissions.data, "dask"))
        computed_emissions = emissions.compute()
        self.assertFalse(np.isnan(computed_emissions).any())
        self.assertTrue((computed_emissions.values >= 0).all())


if __name__ == "__main__":
    unittest.main()
