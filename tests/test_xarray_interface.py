import pytest
import numpy as np
import xarray as xr
import dask.array as da
from pyfengsha.xarray_interface import (
    DustEmissionFENGSHA_xr,
    DustEmissionGOCART2G_xr,
)

# --- Constants for Test Dimensions ---
NI, NJ, NBINS = 10, 20, 3
CHUNK_SIZE = {"lat": 5, "lon": 10}


@pytest.fixture(scope="module")
def common_coords() -> dict:
    """Pytest fixture for common coordinates used in test datasets."""
    return {
        "lat": np.arange(NI),
        "lon": np.arange(NJ),
        "bin": np.arange(NBINS),
    }


@pytest.fixture
def fengsha_dataset(common_coords: dict) -> xr.Dataset:
    """Pytest fixture to create a sample xarray.Dataset for FENGSHA testing."""
    dist_data = np.array([0.1, 0.2, 0.7])
    ds = xr.Dataset(
        {
            "fraclake": (("lat", "lon"), da.zeros((NI, NJ), chunks=CHUNK_SIZE)),
            "fracsnow": (("lat", "lon"), da.zeros((NI, NJ), chunks=CHUNK_SIZE)),
            "oro": (("lat", "lon"), da.ones((NI, NJ), chunks=CHUNK_SIZE)),
            "slc": (("lat", "lon"), da.full((NI, NJ), 0.2, chunks=CHUNK_SIZE)),
            "clay": (("lat", "lon"), da.full((NI, NJ), 0.1, chunks=CHUNK_SIZE)),
            "sand": (("lat", "lon"), da.full((NI, NJ), 0.8, chunks=CHUNK_SIZE)),
            "ssm": (("lat", "lon"), da.full((NI, NJ), 0.9, chunks=CHUNK_SIZE)),
            "rdrag": (("lat", "lon"), da.full((NI, NJ), 0.95, chunks=CHUNK_SIZE)),
            "airdens": (("lat", "lon"), da.full((NI, NJ), 1.2, chunks=CHUNK_SIZE)),
            "ustar": (("lat", "lon"), da.full((NI, NJ), 0.4, chunks=CHUNK_SIZE)),
            "vegfrac": (("lat", "lon"), da.full((NI, NJ), 0.1, chunks=CHUNK_SIZE)),
            "lai": (("lat", "lon"), da.full((NI, NJ), 0.2, chunks=CHUNK_SIZE)),
            "uthrs": (("lat", "lon"), da.full((NI, NJ), 0.25, chunks=CHUNK_SIZE)),
            "distribution": (("bin",), da.from_array(dist_data, chunks=(NBINS,))),
        },
        coords=common_coords,
    )
    ds.attrs["history"] = "2023-01-01T00:00:00Z: Initial dataset creation."
    return ds


@pytest.fixture
def gocart2g_dataset(common_coords: dict) -> xr.Dataset:
    """Pytest fixture to create a sample xarray.Dataset for GOCART2G testing."""
    radius_data = np.array([0.1, 0.5, 1.0])
    ds = xr.Dataset(
        {
            "radius": (("bin",), da.from_array(radius_data, chunks=(NBINS,))),
            "fraclake": (("lat", "lon"), da.zeros((NI, NJ), chunks=CHUNK_SIZE)),
            "gwettop": (("lat", "lon"), da.full((NI, NJ), 0.1, chunks=CHUNK_SIZE)),
            "oro": (("lat", "lon"), da.ones((NI, NJ), chunks=CHUNK_SIZE)),
            "u10m": (("lat", "lon"), da.full((NI, NJ), 5.0, chunks=CHUNK_SIZE)),
            "v10m": (("lat", "lon"), da.full((NI, NJ), 2.0, chunks=CHUNK_SIZE)),
            "du_src": (("lat", "lon"), da.ones((NI, NJ), chunks=CHUNK_SIZE)),
        },
        coords=common_coords,
    )
    ds.attrs["history"] = "2023-01-01T00:00:00Z: Initial GOCART2G dataset."
    return ds


def test_fengsha_dataset_input_and_provenance(fengsha_dataset: xr.Dataset):
    """Test FENGSHA with a Dataset input and check for provenance."""
    emissions = DustEmissionFENGSHA_xr(
        ds=fengsha_dataset,
        alpha=1.0,
        gamma=1.0,
        kvhmax=2.0e-4,
        grav=9.81,
        drylimit_factor=1.0,
        moist_correct=1.0,
        drag_opt=1,
    )
    assert isinstance(emissions, xr.DataArray)
    assert emissions.shape == (NI, NJ, NBINS)
    assert "history" in emissions.attrs
    assert "FENGSHA scheme" in emissions.attrs["history"]
    assert hasattr(emissions.data, "dask")

    computed_emissions = emissions.compute()
    assert not np.isnan(computed_emissions).any()


def test_gocart2g_dataset_input_and_provenance(gocart2g_dataset: xr.Dataset):
    """Test GOCART2G with a Dataset input and check provenance."""
    emissions = DustEmissionGOCART2G_xr(ds=gocart2g_dataset, Ch_DU=1.0e-5, grav=9.81)

    # 1. Verify Output Type, Shape, and Dims
    assert isinstance(emissions, xr.DataArray)
    assert emissions.shape == (NI, NJ, NBINS)
    assert emissions.dims == ("lat", "lon", "bin")

    # 2. Verify Coordinates are Preserved
    xr.testing.assert_equal(emissions.coords.to_dataset(), gocart2g_dataset.coords.to_dataset())

    # 3. Verify Provenance (History) Tracking
    assert "history" in emissions.attrs
    assert emissions.attrs["history"].startswith("20")
    assert "GOCART2G scheme" in emissions.attrs["history"]
    assert gocart2g_dataset.attrs["history"] in emissions.attrs["history"]

    # 4. Verify Computation (Lazy Execution and Valid Values)
    assert hasattr(emissions.data, "dask")
    computed_emissions = emissions.compute()
    assert not np.isnan(computed_emissions).any()
    assert (computed_emissions.values >= 0).all()
