# FENGSHA Dust Emission Scheme

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

This package provides a pure Python implementation (accelerated with Numba) of the NOAA/ARL FENGSHA dust emission model and the GOCART2G scheme. It is designed to be efficient and easy to integrate with xarray-based workflows.

## Features

*   **Pure Python**: No Fortran compilation required, avoiding complex build steps.
*   **High Performance**: Uses [Numba](https://numba.pydata.org/) for JIT compilation, offering performance comparable to Fortran.
*   **Xarray Integration**: Built-in wrappers for [xarray](https://xarray.pydata.org/) `apply_ufunc`, allowing easy application over large datasets with Dask support.
*   **Multiple Schemes**: Includes both FENGSHA and GOCART2G implementations.

## Installation

The easiest way to install is using pip from the source directory:

```bash
git clone https://github.com/yourusername/pyfengsha.git
cd pyfengsha
pip install .
```

To install with dependencies for running examples and building documentation:

```bash
pip install .[docs,examples]
```

## Quick Start

Here is a simple example of how to use the xarray interface:

```python
import xarray as xr
import pyfengsha

# Load your data (example)
# ds = xr.open_dataset("your_data.nc")

# Define parameters
alpha = 1.0
gamma = 1.0
kvhmax = 1.0
grav = 9.81
drylimit_factor = 1.0
moist_correct = 1.0
drag_opt = 1

# Call the FENGSHA xarray wrapper
# Note: Ensure your input DataArrays have the correct dimensions (lat, lon) or (bin)
emissions = pyfengsha.DustEmissionFENGSHA_xr(
    fraclake=ds.fraclake,
    fracsnow=ds.fracsnow,
    oro=ds.oro,
    slc=ds.slc,
    clay=ds.clay,
    sand=ds.sand,
    silt=ds.silt,
    ssm=ds.ssm,
    rdrag=ds.rdrag,
    airdens=ds.airdens,
    ustar=ds.ustar,
    vegfrac=ds.vegfrac,
    lai=ds.lai,
    uthrs=ds.uthrs,
    alpha=alpha,
    gamma=gamma,
    kvhmax=kvhmax,
    grav=grav,
    rhop=ds.rhop,
    distribution=ds.distribution,
    drylimit_factor=drylimit_factor,
    moist_correct=moist_correct,
    drag_opt=drag_opt
)
```

See the [Documentation](https://yourusername.github.io/pyfengsha/) for more detailed examples and API reference.

## References

If you use this code, please consider citing the following:

**FENGSHA Model:**
*   Tong, D., Dan, M., Wang, T., & Lee, P. (2017). Long-term dust climatology in the western United States reconstructed from satellite observations, ground measurements and model simulations. *Aeolian Research*, 24, 103-115.
*   Developed at NOAA ARL by Daniel Tong and Dale Gillette for NAQFC (2012).

**GOCART Scheme:**
*   Ginoux, P., Chin, M., Tegen, I., Prospero, J. M., Holben, B., Dubovik, O., & Lin, S. J. (2001). Sources and distributions of dust aerosols simulated with the GOCART model. *Journal of Geophysical Research: Atmospheres*, 106(D17), 20255-20273.
*   LeGrand, S. L., Polashenski, C., Letcher, T. W., Creighton, G. A., Peckham, S. E., and Cetola, J. D. (2019). The AFWA dust emission scheme for the GOCART aerosol model in WRF-Chem v3.8.1. *Geosci. Model Dev.*, 12, 131–166.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
