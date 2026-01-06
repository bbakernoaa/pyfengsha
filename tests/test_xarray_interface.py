import xarray as xr
import numpy as np
import pytest
from pyfengsha import DustEmissionFENGSHA_xr, DustEmissionGOCART2G_xr


@pytest.fixture
def sample_fengsha_dataset():
    """Creates a sample FENGSHA xarray.Dataset for testing."""
    return xr.Dataset(
        {
            "fraclake": (("lat", "lon"), np.zeros((10, 20))),
            "fracsnow": (("lat", "lon"), np.zeros((10, 20))),
            "oro": (("lat", "lon"), np.ones((10, 20))),
            "slc": (("lat", "lon"), np.full((10, 20), 0.2)),
            "clay": (("lat", "lon"), np.full((10, 20), 0.1)),
            "sand": (("lat", "lon"), np.full((10, 20), 0.8)),
            "ssm": (("lat", "lon"), np.full((10, 20), 0.9)),
            "rdrag": (("lat", "lon"), np.full((10, 20), 0.95)),
            "airdens": (("lat", "lon"), np.full((10, 20), 1.2)),
            "ustar": (("lat", "lon"), np.full((10, 20), 0.4)),
            "vegfrac": (("lat", "lon"), np.full((10, 20), 0.1)),
            "lai": (("lat", "lon"), np.full((10, 20), 0.2)),
            "uthrs": (("lat", "lon"), np.full((10, 20), 0.25)),
            "distribution": (("bin",), np.array([0.1, 0.2, 0.7])),
        },
        coords={"lat": np.arange(10), "lon": np.arange(20), "bin": np.arange(3)},
    )


@pytest.fixture
def sample_gocart_dataset():
    """Creates a sample GOCART2G xarray.Dataset for testing."""
    return xr.Dataset(
        {
            "radius": (("bin",), np.array([0.1, 0.5, 1.0])),
            "fraclake": (("lat", "lon"), np.zeros((10, 20))),
            "gwettop": (("lat", "lon"), np.full((10, 20), 0.1)),
            "oro": (("lat", "lon"), np.ones((10, 20))),
            "u10m": (("lat", "lon"), np.full((10, 20), 5.0)),
            "v10m": (("lat", "lon"), np.full((10, 20), 2.0)),
            "du_src": (("lat", "lon"), np.ones((10, 20))),
        },
        coords={"lat": np.arange(10), "lon": np.arange(20), "bin": np.arange(3)},
    )


def test_dust_emission_fengsha_xr_smoke(sample_fengsha_dataset):
    """Smoke test for the DustEmissionFENGSHA_xr wrapper."""
    ds = sample_fengsha_dataset
    emissions = DustEmissionFENGSHA_xr(
        ds=ds.chunk({"lat": 5, "lon": 10}),
        alpha=1.0,
        gamma=1.0,
        kvhmax=2.0e-4,
        grav=9.81,
        drylimit_factor=1.0,
        moist_correct=1.0,
        drag_opt=1,
    )

    assert emissions.shape == (10, 20, 3)
    assert emissions.dims == ("lat", "lon", "bin")
    assert "history" in emissions.attrs
    assert hasattr(emissions.data, "dask")
    assert not np.isnan(emissions.compute()).all()


def test_dust_emission_fengsha_xr_custom_dims(sample_fengsha_dataset):
    """Tests if DustEmissionFENGSHA_xr works with custom dimension names."""
    ds = sample_fengsha_dataset.rename({"lat": "y", "lon": "x", "bin": "particle_bin"})
    custom_dims = {"lat": "y", "lon": "x", "bin": "particle_bin"}

    emissions = DustEmissionFENGSHA_xr(
        ds=ds.chunk({"y": 5, "x": 10}),
        core_dims_mapping=custom_dims,
        alpha=1.0,
        gamma=1.0,
        kvhmax=2.0e-4,
        grav=9.81,
        drylimit_factor=1.0,
        moist_correct=1.0,
        drag_opt=1,
    )

    assert emissions.shape == (10, 20, 3)
    assert emissions.dims == ("y", "x", "particle_bin")
    assert hasattr(emissions.data, "dask")
    assert not np.isnan(emissions.compute()).all()


def test_dust_emission_gocart2g_xr_custom_dims(sample_gocart_dataset):
    """Tests if DustEmissionGOCART2G_xr works with custom dimension names."""
    ds = sample_gocart_dataset.rename({"lat": "y", "lon": "x", "bin": "particle_bin"})
    custom_dims = {"lat": "y", "lon": "x", "bin": "particle_bin"}

    emissions = DustEmissionGOCART2G_xr(
        ds=ds.chunk({"y": 5, "x": 10}),
        Ch_DU=1.0e-9,
        grav=9.81,
        core_dims_mapping=custom_dims,
    )

    assert emissions.shape == (10, 20, 3)
    assert emissions.dims == ("y", "x", "particle_bin")
    assert hasattr(emissions.data, "dask")
    assert not np.isnan(emissions.compute()).all()
