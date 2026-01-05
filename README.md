# FENGSHA Dust emission scheme

This is a wrapper around the fortran dust emission scheme.

### To Install

The easiest way to install is using pip

```cd``` into the pyfengsha directory where the setup.py file exists

Then execute the command:

```
pip install .
```

### To use

Simple example on how to use.  First import pyfengsha

```python

import pyfengsha

```

now all subroutines in the fengsha.F90 file is available to you in python

```python
Help on package pyfengsha:

NAME
    pyfengsha

PACKAGE CONTENTS
    fengsha

FUNCTIONS
    draxler_hflux(ustar, threshold_velocity)

    fecan_dry_limit(clayfrac)

    fecan_moisture_correction(volumetric_soil_moisture, sandfrac, clayfrac)

    fengsha(rhoa, volumetric_soil_moisture, ssm, land, ustar, clayfrac, sandfrac, drag_partition, dry_threshold)

    fengsha_albedo(rhoa, volumetric_soil_moisture, ssm, land, ustar, clayfrac, sandfrac, drag_partition, dry_threshold)

    mackinnon_drag_partition(z0)

    mb95_drag_partition(z0)

    mb95_kvh(clay)

    modified_threshold_velocity(dry_threshold, moisture_correction, drag_partition)

    shao_1996_soil_moisture(volumetric_soil_moisture)

    shao_2004_soil_moisture(volumetric_soil_moisture)

    volumetric_to_gravimetric(volumetric_soil_moisture, sandfrac)

```


The `pyfengsha` package now includes a user-friendly Xarray wrapper, `DustEmissionFENGSHA_xr`, which handles the `apply_ufunc` boilerplate for you. This is the recommended way to apply the FENGSHA scheme to gridded data.

The function takes a single `xarray.Dataset` containing all the required input variables as `DataArrays`.

Here is an example of how to use it. First, create a sample `xarray.Dataset` (replace this with your actual data):

```python
import xarray as xr
import numpy as np
import pyfengsha

# Create a sample dataset (replace with real data)
ds = xr.Dataset(
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

# Run the FENGSHA model using the Xarray wrapper
# Use dask for lazy evaluation on large datasets
emissions = pyfengsha.DustEmissionFENGSHA_xr(
    ds=ds.chunk({'lat': 5, 'lon': 10}),
    alpha=1.0,
    gamma=1.0,
    kvhmax=2.0E-4,
    grav=9.81,
    drylimit_factor=1.0,
    moist_correct=1.0,
    drag_opt=1
)

print(emissions)
```

