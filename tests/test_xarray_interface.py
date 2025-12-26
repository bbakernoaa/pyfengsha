import unittest
import numpy as np
import xarray as xr
import dask.array as da
from pyfengsha.xarray_interface import DustEmissionFENGSHA_xr

class TestXarrayInterface(unittest.TestCase):

    def setUp(self):
        """Set up a sample xarray.Dataset for testing."""
        self.ni, self.nj, self.nbins = 10, 20, 3
        self.coords = {
            'lat': np.arange(self.ni),
            'lon': np.arange(self.nj),
            'bin': np.arange(self.nbins)
        }
        # Use Dask arrays to test lazy evaluation
        # Explicitly create a numpy array before wrapping with dask
        dist_data = np.array([0.1, 0.2, 0.7])
        self.ds = xr.Dataset({
            'fraclake': (('lat', 'lon'), da.zeros((self.ni, self.nj), chunks=(5, 10))),
            'fracsnow': (('lat', 'lon'), da.zeros((self.ni, self.nj), chunks=(5, 10))),
            'oro': (('lat', 'lon'), da.ones((self.ni, self.nj), chunks=(5, 10))),
            'slc': (('lat', 'lon'), da.full((self.ni, self.nj), 0.2, chunks=(5, 10))),
            'clay': (('lat', 'lon'), da.full((self.ni, self.nj), 0.1, chunks=(5, 10))),
            'sand': (('lat', 'lon'), da.full((self.ni, self.nj), 0.8, chunks=(5, 10))),
            'ssm': (('lat', 'lon'), da.full((self.ni, self.nj), 0.9, chunks=(5, 10))),
            'rdrag': (('lat', 'lon'), da.full((self.ni, self.nj), 0.95, chunks=(5, 10))),
            'airdens': (('lat', 'lon'), da.full((self.ni, self.nj), 1.2, chunks=(5, 10))),
            'ustar': (('lat', 'lon'), da.full((self.ni, self.nj), 0.4, chunks=(5, 10))),
            'vegfrac': (('lat', 'lon'), da.full((self.ni, self.nj), 0.1, chunks=(5, 10))),
            'lai': (('lat', 'lon'), da.full((self.ni, self.nj), 0.2, chunks=(5, 10))),
            'uthrs': (('lat', 'lon'), da.full((self.ni, self.nj), 0.25, chunks=(5, 10))),
            'distribution': (('bin',), da.from_array(dist_data, chunks=(self.nbins,)))
        }, coords=self.coords)
        self.ds.attrs['history'] = '2023-01-01T00:00:00Z: Initial dataset creation.'

    def test_dataset_input_and_provenance(self):
        """Test the refactored function with a Dataset input and check for provenance."""
        emissions = DustEmissionFENGSHA_xr(
            ds=self.ds,
            alpha=1.0,
            gamma=1.0,
            kvhmax=2.0E-4,
            grav=9.81,
            drylimit_factor=1.0,
            moist_correct=1.0,
            drag_opt=1
        )

        # 1. Verify Output Type and Shape
        self.assertIsInstance(emissions, xr.DataArray)
        self.assertEqual(emissions.shape, (self.ni, self.nj, self.nbins))
        self.assertEqual(emissions.dims, ('lat', 'lon', 'bin'))

        # 2. Verify Coordinates are Preserved by comparing to a dummy array
        # with the expected coordinates. xr.testing.assert_allclose implicitly
        # checks coordinates.
        dummy_array = xr.DataArray(
            da.zeros((self.ni, self.nj, self.nbins), chunks=(5, 10, self.nbins)),
            coords=self.ds.coords,
            dims=('lat', 'lon', 'bin')
        )
        xr.testing.assert_allclose(emissions.coords.to_dataset(), dummy_array.coords.to_dataset())


        # 3. Verify Provenance (History) Tracking
        self.assertIn('history', emissions.attrs)
        self.assertTrue(emissions.attrs['history'].startswith('20')) # Starts with a timestamp
        self.assertIn('FENGSHA scheme', emissions.attrs['history'])
        self.assertIn(self.ds.attrs['history'], emissions.attrs['history']) # Original history is preserved

        # 4. Verify Computation (Lazy Execution)
        # Check that the result is a Dask array before computation
        self.assertTrue(hasattr(emissions.data, 'dask'))
        # Compute the result and check that values are valid
        computed_emissions = emissions.compute()
        self.assertFalse(np.isnan(computed_emissions).any())
        self.assertTrue((computed_emissions.values >= 0).all())

if __name__ == '__main__':
    unittest.main()
