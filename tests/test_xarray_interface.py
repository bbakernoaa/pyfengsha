import xarray as xr
import numpy as np
import pytest
from pyfengsha import DustEmissionFENGSHA_xr


@pytest.fixture
def sample_dataset():
    """Creates a sample xarray.Dataset for testing."""
    return xr.Dataset(
        {
            'fraclake': (('lat', 'lon'), np.zeros((10, 20))),
            'fracsnow': (('lat', 'lon'), np.zeros((10, 20))),
            'oro': (('lat', 'lon'), np.ones((10, 20))),
            'slc': (('lat', 'lon'), np.full((10, 20), 0.2)),
            'clay': (('lat', 'lon'), np.full((10, 20), 0.1)),
            'sand': (('lat', 'lon'), np.full((10, 20), 0.8)),
            'ssm': (('lat', 'lon'), np.full((10, 20), 0.9)),
            'rdrag': (('lat', 'lon'), np.full((10, 20), 0.95)),
            'airdens': (('lat', 'lon'), np.full((10, 20), 1.2)),
            'ustar': (('lat', 'lon'), np.full((10, 20), 0.4)),
            'vegfrac': (('lat', 'lon'), np.full((10, 20), 0.1)),
            'lai': (('lat', 'lon'), np.full((10, 20), 0.2)),
            'uthrs': (('lat', 'lon'), np.full((10, 20), 0.25)),
            'distribution': (('bin',), np.array([0.1, 0.2, 0.7]))
        },
        coords={'lat': np.arange(10), 'lon': np.arange(20), 'bin': np.arange(3)}
    )


def test_dust_emission_fengsha_xr_smoke(sample_dataset):
    """
    Smoke test for the DustEmissionFENGSHA_xr wrapper.

    Checks for:
    - Output shape and dimensions
    - Coordinate preservation
    - History attribute addition
    - Dask compatibility
    """
    ds = sample_dataset
    # Run the FENGSHA model with Dask
    emissions = DustEmissionFENGSHA_xr(
        ds=ds.chunk({'lat': 5, 'lon': 10}),
        alpha=1.0,
        gamma=1.0,
        kvhmax=2.0E-4,
        grav=9.81,
        drylimit_factor=1.0,
        moist_correct=1.0,
        drag_opt=1
    )

    # 1. The Proof (Validation)
    # Check output shape and dimensions
    assert emissions.shape == (10, 20, 3)
    assert emissions.dims == ('lat', 'lon', 'bin')

    # Check coordinate preservation
    xr.testing.assert_equal(emissions.coords['lat'], ds.coords['lat'])
    xr.testing.assert_equal(emissions.coords['lon'], ds.coords['lon'])
    xr.testing.assert_equal(emissions.coords['bin'], ds.coords['bin'])

    # Check for history attribute
    assert "history" in emissions.attrs
    assert "FENGSHA scheme" in emissions.attrs["history"]

    # Check that the result is a Dask array
    assert hasattr(emissions.data, 'dask')

    # Trigger computation and check for basic numeric validity
    computed_emissions = emissions.compute()
    assert not np.isnan(computed_emissions).all()


def test_dust_emission_fengsha_xr_custom_dims(sample_dataset):
    """
    Tests the core_dims_mapping functionality.
    Verifies that the wrapper can handle non-standard dimension names
    and that the output dimensions match the original input dimensions.
    """
    # Rename dimensions to something non-standard
    ds_custom = sample_dataset.rename(
        {'lat': 'y', 'lon': 'x', 'bin': 'particle_size'}
    )

    # Define the mapping from the custom names to the internal names
    mapping = {'y': 'lat', 'x': 'lon', 'particle_size': 'bin'}

    # Run the FENGSHA model with the mapping
    emissions = DustEmissionFENGSHA_xr(
        ds=ds_custom.chunk({'y': 5, 'x': 10}),
        alpha=1.0,
        gamma=1.0,
        kvhmax=2.0E-4,
        grav=9.81,
        drylimit_factor=1.0,
        moist_correct=1.0,
        drag_opt=1,
        core_dims_mapping=mapping
    )

    # 1. The Proof (Validation)
    # Check that the output dimensions are the original, custom names
    assert emissions.dims == ('y', 'x', 'particle_size')

    # Check coordinate preservation with custom names
    xr.testing.assert_equal(emissions.coords['y'], ds_custom.coords['y'])
    xr.testing.assert_equal(emissions.coords['x'], ds_custom.coords['x'])
    xr.testing.assert_equal(
        emissions.coords['particle_size'], ds_custom.coords['particle_size']
    )

    # Check shape
    assert emissions.shape == (10, 20, 3)

    # Check that it's still a dask array
    assert hasattr(emissions.data, 'dask')

    # Trigger computation and check for basic numeric validity
    computed_emissions = emissions.compute()
    assert not np.isnan(computed_emissions).all()
