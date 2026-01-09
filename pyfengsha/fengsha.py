"""
NOAA/ARL FENGSHA dust emission model and GOCART2G scheme implementations.
"""

import math

import numpy as np
from numba import jit, vectorize
from numpy.typing import NDArray

# --- Constants ---
# Using descriptive names for physical and model constants.
# No more magic numbers.

# Physical Constants
G_ACCEL_CMS2: float = 9.81 * 100.0  # Gravity in cm/s^2

# Soil and Water Properties
SOIL_DENSITY_GCM3: float = 2.65
WATER_DENSITY_GCM3: float = 1.0
GOCART_PARTICLE_DENSITY_GCM3: float = 1.7

# Fecan Moisture Correction Parameters
FECAN_CLAY_COEFF_A: float = 14.0
FECAN_CLAY_COEFF_B: float = 17.0
FECAN_MOISTURE_COEFF_A: float = 1.21
FECAN_MOISTURE_COEFF_B: float = 0.68

# Drag Partition Scheme Constants (Darmenova)
DRAG_MIN_VAL: float = 1.0e-3
SIGB_D: float = 1.0
MB_D: float = 0.5
BETAB_D: float = 90.0
SIGV_D: float = 1.45
MV_D: float = 0.16
BETAV_D: float = 202.0

# Drag Partition Scheme Constants (Leung)
LAI_THRESHOLD_L: float = 0.33
C_L: float = 4.8
F0_L: float = 0.32
SIGB_L: float = 1.0
MB_L: float = 0.5
BETAB_L: float = 90.0
MIN_FEFF_L: float = 1.0e-5
MAX_FEFF_L: float = 1.0

# Thresholds and masks for main emission calculation
SSM_THRESHOLD: float = 1.0e-02
VEG_THRESHOLD_FENGSHA: float = 0.4
LAND_MASK_VALUE: float = 1.0
MAX_RDRAG_FENGSHA: float = 0.3

# --- Helper Functions ---


@jit(nopython=True)
def volumetric_to_gravimetric(vsoil: float, sandfrac: float) -> float:
    """Convert volumetric soil moisture to gravimetric based on FENGSHA scheme.

    Parameters
    ----------
    vsoil : float
        Volumetric Soil Moisture [m3/m3].
    sandfrac : float
        Fractional Sand content [0-1].

    Returns
    -------
    float
        Gravimetric soil moisture [kg/kg].
    """
    vsat = 0.489 - 0.00126 * (sandfrac * 100.0)
    return (
        vsoil
        * WATER_DENSITY_GCM3
        * 1000.0
        / (SOIL_DENSITY_GCM3 * 1000.0 * (1.0 - vsat))
    )


@jit(nopython=True)
def gocart_vol_to_grav(vsoil: float, sandfrac: float) -> float:
    """Convert volumetric soil moisture to gravimetric based on GOCART scheme.

    Note
    ----
    The original Fortran code returned a value scaled by 100. This is
    now done at the call site for clarity.

    Parameters
    ----------
    vsoil : float
        Volumetric Soil Moisture [m3/m3].
    sandfrac : float
        Fractional Sand content [0-1].

    Returns
    -------
    float
        Gravimetric soil moisture [kg/kg].
    """
    vsat = 0.489 - 0.126 * sandfrac
    return (
        vsoil
        * WATER_DENSITY_GCM3
        * 1000.0
        / (GOCART_PARTICLE_DENSITY_GCM3 * 1000.0 * (1.0 - vsat))
    )


@jit(nopython=True)
def fecan_dry_limit(clay: float) -> float:
    """Calculates the Fecan dry limit for soil moisture.

    Parameters
    ----------
    clay : float
        Clay fraction [0-1].

    Returns
    -------
    float
        The Fecan dry limit for soil moisture.
    """
    if clay <= 0.0:
        # Using an epsilon for stability with zero clay
        clay = 1.0e-4
    return FECAN_CLAY_COEFF_A * clay * clay + FECAN_CLAY_COEFF_B * clay


@jit(nopython=True)
def fecan_moisture_correction(
    vol_soil_moisture: float, sand: float, clay: float
) -> float:
    """Calculates the Fecan soil moisture correction factor (H).

    Parameters
    ----------
    vol_soil_moisture : float
        Volumetric soil moisture [m3/m3].
    sand : float
        Sand fraction [0-1].
    clay : float
        Clay fraction [0-1].

    Returns
    -------
    float
        The Fecan soil moisture correction factor.
    """
    gravsm = volumetric_to_gravimetric(vol_soil_moisture, sand)
    drylimit = fecan_dry_limit(clay)

    if gravsm > drylimit:
        return np.sqrt(
            1.0 + FECAN_MOISTURE_COEFF_A * (gravsm - drylimit) ** FECAN_MOISTURE_COEFF_B
        )
    return 1.0


@jit(nopython=True)
def gocart_moisture_correction(slc: float, sand: float, clay: float, b: float) -> float:
    """
    Calculates the GOCART version of the soil moisture correction factor.

    Parameters
    ----------
    slc : float
        Soil liquid content.
    sand : float
        Sand fraction [0-1].
    clay : float
        Clay fraction [0-1].
    b : float
        A tuning factor for the dry limit calculation.

    Returns
    -------
    float
        The calculated moisture correction factor.
    """
    # Note: GOCART scales gravimetric moisture by 100.
    gravimetric_soil_moisture = gocart_vol_to_grav(slc, sand) * 100.0
    fecan_dry_limit_val = b * clay * (FECAN_CLAY_COEFF_A * clay + FECAN_CLAY_COEFF_B)

    return np.sqrt(
        1.0 + 1.21 * max(0.0, gravimetric_soil_moisture - fecan_dry_limit_val) ** 0.68
    )


@jit(nopython=True)
def shao_1996_soil_moisture(w: float) -> float:
    """
    Calculates the Shao 1996 soil moisture function.

    Parameters
    ----------
    w : float
        Gravimetric soil moisture [kg/kg].

    Returns
    -------
    float
        The calculated soil moisture factor.
    """
    return np.exp(22.7 * w)


@jit(nopython=True)
def shao_2004_soil_moisture(w: float) -> float:
    """
    Calculates the Shao 2004 soil moisture function.

    Parameters
    ----------
    w : float
        Gravimetric soil moisture [kg/kg].

    Returns
    -------
    float
        The calculated soil moisture factor.
    """
    if w <= 0.03:
        return np.exp(22.7 * w)
    return np.exp(95.3 * w - 2.029)


@jit(nopython=True)
def modified_threshold_velocity(u_ts0: float, H: float, drag: float) -> float:
    """
    Calculates the modified threshold velocity.

    Parameters
    ----------
    u_ts0 : float
        Threshold velocity over an ideal surface.
    H : float
        Soil moisture correction factor.
    drag : float
        Drag partition factor.

    Returns
    -------
    float
        The modified threshold velocity.
    """
    return u_ts0 * H / drag


@jit(nopython=True)
def mb95_vertical_flux_ratio(clay: float, max_ratio: float = 2.0e-4) -> float:
    """
    Calculates the Marticorena and Bergametti (1995) vertical flux ratio.

    Parameters
    ----------
    clay : float
        Clay fraction [0-1].
    max_ratio : float, optional
        The maximum allowed ratio, by default 2.0E-4.

    Returns
    -------
    float
        The vertical to horizontal flux ratio.
    """
    if clay <= 0.2:
        return 10.0 ** (13.4 * clay - 6.0)
    return max_ratio


@jit(nopython=True)
def horizontal_saltation_flux(ust: float, utst: float) -> float:
    """
    Calculates the Horizontal Saltation Flux (Q).

    Parameters
    ----------
    ust : float
        Friction velocity.
    utst : float
        Modified threshold velocity.

    Returns
    -------
    float
        The horizontal saltation flux. Returns 0 if ust <= utst.
    """
    if ust <= utst:
        return 0.0
    return ust * (ust * ust - utst * utst)


@jit(nopython=True)
def mackinnon_drag_partition(z0: float) -> float:
    """
    Calculates the MacKinnon drag partition scheme.

    Parameters
    ----------
    z0 : float
        Roughness length.

    Returns
    -------
    float
        The calculated drag partition factor.
    """
    z0s = 1.0e-04
    return 1.0 - np.log(z0 / z0s) / np.log(0.7 * (12255.0 / z0s) ** 0.8)


@jit(nopython=True)
def mb95_drag_partition(z0: float) -> float:
    """
    Calculates the Marticorena and Bergametti (1995) drag partition scheme.

    Parameters
    ----------
    z0 : float
        Roughness length.

    Returns
    -------
    float
        The calculated drag partition factor.
    """
    z0s = 1.0e-04
    return 1.0 - np.log(z0 / z0s) / np.log(0.7 * (10.0 / z0s) ** 0.8)


@vectorize(["float64(float64, float64, float64)"])
def _kok_aerosol_distribution_ufunc(
    radius: float, r_low: float, r_up: float
) -> float:
    """
    Numba ufunc for element-wise Kok aerosol distribution calculation.

    Parameters
    ----------
    radius : float
        Particle radius for the bin [m].
    r_low : float
        Lower bound radius for the bin [m].
    r_up : float
        Upper bound radius for the bin [m].

    Returns
    -------
    float
        The un-normalized volume for the bin.
    """
    median_mass_diameter = 3.4
    geom_std_dev = 3.0
    crack_prop_len = 12.0
    factor = 1.0 / (math.sqrt(2.0) * math.log(geom_std_dev))

    diameter = 2.0 * radius
    dlam = diameter / crack_prop_len

    erf_arg = factor * math.log(diameter / median_mass_diameter)

    return (
        diameter
        * (1.0 + math.erf(erf_arg))
        * math.exp(-(dlam**3))
        * math.log(r_up / r_low)
    )


def kok_aerosol_distribution(radius: NDArray, r_low: NDArray, r_up: NDArray) -> NDArray:
    """
    Computes Kok's dust size aerosol distribution (Numba-vectorized).

    This function calculates the volume distribution of aerosols across a set
    of size bins based on Kok's model. The implementation uses a Numba
    vectorized ufunc for high performance.

    Parameters
    ----------
    radius : NDArray
        1D array of particle radii for each bin [m].
    r_low : NDArray
        1D array of the lower bound radius for each bin [m].
    r_up : NDArray
        1D array of the upper bound radius for each bin [m].

    Returns
    -------
    NDArray
        1D array of the normalized volume distribution for each bin (unitless).

    Examples
    --------
    >>> radius = np.array([0.1, 0.5, 1.0])
    >>> r_low = np.array([0.05, 0.45, 0.95])
    >>> r_up = np.array([0.15, 0.55, 1.05])
    >>> result = kok_aerosol_distribution(radius, r_low, r_up)
    >>> # The result should be close to the original SciPy implementation
    >>> expected = np.array([0.16568854, 0.39523267, 0.43907879])
    >>> np.allclose(result, expected)
    True
    """
    # Call the high-performance Numba ufunc for the core calculation
    distribution = _kok_aerosol_distribution_ufunc(radius, r_low, r_up)

    total_volume = np.sum(distribution)

    # Avoid division by zero if total_volume is zero
    if total_volume > 1.0e-15:
        return distribution / total_volume
    return np.zeros_like(distribution)


# --- Main Emission Schemes ---


@jit(nopython=True)
def fengsha(
    rho_phy: float,
    smois: float,
    ssm: float,
    xland: float,
    ust: float,
    clay: float,
    sand: float,
    rdrag: float,
    u_ts0: float,
) -> float:
    """
    Calculates the core FENGSHA dust emission for a single grid cell.

    Parameters
    ----------
    rho_phy : float
        Air density.
    smois : float
        Soil moisture.
    ssm : float
        Surface soil moisture.
    xland : float
        Land mask (1 for land).
    ust : float
        Friction velocity.
    clay : float
        Clay fraction [0-1].
    sand : float
        Sand fraction [0-1].
    rdrag : float
        Drag partition parameter.
    u_ts0 : float
        Threshold velocity over an ideal surface.

    Returns
    -------
    float
        The calculated dust emission flux for a single cell.
    """
    if xland != LAND_MASK_VALUE or ssm <= 0:
        return 0.0

    H = fecan_moisture_correction(smois, sand, clay)
    kvh = mb95_vertical_flux_ratio(clay)
    u_ts = modified_threshold_velocity(u_ts0, H, rdrag)
    Q = horizontal_saltation_flux(ust, u_ts)

    return ssm * rho_phy / G_ACCEL_CMS2 * kvh * Q


@jit(nopython=True)
def fengsha_albedo(
    rho_phy: float,
    smois: float,
    ssm: float,
    xland: float,
    ust: float,
    clay: float,
    sand: float,
    rdrag: float,
    u_ts0: float,
) -> float:
    """
    Calculates dust emission using FENGSHA albedo-based logic for a single cell.

    Parameters
    ----------
    rho_phy : float
        Air density.
    smois : float
        Soil moisture.
    ssm : float
        Surface soil moisture.
    xland : float
        Land mask (1 for land).
    ust : float
        Friction velocity.
    clay : float
        Clay fraction [0-1].
    sand : float
        Sand fraction [0-1].
    rdrag : float
        Drag partition parameter.
    u_ts0 : float
        Threshold velocity over an ideal surface.

    Returns
    -------
    float
        The calculated dust emission flux.
    """
    if xland != LAND_MASK_VALUE or ssm <= 0:
        return 0.0

    H = fecan_moisture_correction(smois, sand, clay)
    kvh = mb95_vertical_flux_ratio(clay)
    u_ts = modified_threshold_velocity(u_ts0, H, rdrag)

    ustar_albedo = ust * rdrag
    Q = horizontal_saltation_flux(ustar_albedo, u_ts)

    return ssm * rho_phy / G_ACCEL_CMS2 * kvh * Q


@jit(nopython=True)
def darmenova_drag_partition(Lc: float, vegfrac: float, thresh: float) -> float:
    """
    Calculates the Darmenova drag partition scheme for a single grid cell.

    Parameters
    ----------
    Lc : float
        Aerodynamic roughness length.
    vegfrac : float
        Vegetation fraction.
    thresh : float
        Vegetation fraction threshold.

    Returns
    -------
    float
        The effective drag partition factor.
    """
    if vegfrac < 0.0 or vegfrac >= thresh:
        feff_veg = DRAG_MIN_VAL
    else:
        Lc_veg = -0.35 * np.log(1.0 - vegfrac)
        R1 = 1.0 / np.sqrt(1.0 - SIGV_D * MV_D * Lc_veg)
        R2 = 1.0 / np.sqrt(1.0 + MV_D * BETAV_D * Lc_veg)
        feff_veg = R1 * R2

    Lc_bare = Lc / (1.0 - vegfrac)
    tmpVal = 1.0 - SIGB_D * MB_D * Lc_bare

    if not (vegfrac < 0.0 or vegfrac >= thresh or Lc > 0.2 or tmpVal <= 0.0):
        R1 = 1.0 / np.sqrt(1.0 - SIGB_D * MB_D * Lc_bare)
        R2 = 1.0 / np.sqrt(1.0 + MB_D * BETAB_D * Lc_bare)
        feff_bare = R1 * R2
    else:
        feff_bare = DRAG_MIN_VAL

    feff = feff_veg * feff_bare
    return feff if 1.0e-5 <= feff <= 1.0 else DRAG_MIN_VAL


@jit(nopython=True)
def leung_drag_partition(Lc: float, lai: float, gvf: float, thresh: float) -> float:
    """
    Calculates the Leung drag partition scheme for a single grid cell.

    Parameters
    ----------
    Lc : float
        Aerodynamic roughness length.
    lai : float
        Leaf Area Index.
    gvf : float
        Green vegetation fraction.
    thresh : float
        LAI threshold.

    Returns
    -------
    float
        The effective drag partition factor.
    """
    SMALL_VAL = 1.0e-10
    frac_bare = max(min(1.0 - lai / thresh, 1.0), SMALL_VAL)

    if lai <= 0.0 or lai >= thresh:
        feff_veg = 0.0
    else:
        K = 2.0 * (1.0 / max(1.0 - lai, SMALL_VAL) - 1.0)
        feff_veg = (K + F0_L * C_L) / (K + C_L)

    if 0.0 < Lc <= 0.2 and lai < thresh:
        Lc_bare = Lc / max(frac_bare, SMALL_VAL)
        tmpVal = 1.0 - SIGB_L * MB_L * Lc_bare
        if tmpVal > SMALL_VAL:
            Rbare1 = 1.0 / np.sqrt(max(1.0 - SIGB_L * MB_L * Lc_bare, SMALL_VAL))
            Rbare2 = 1.0 / np.sqrt(1.0 + BETAB_L * MB_L * Lc_bare)
            feff_bare = Rbare1 * Rbare2
        else:
            feff_bare = 0.0
    else:
        feff_bare = 0.0

    feff = (gvf * feff_veg**3 + frac_bare * feff_bare**3) ** (1.0 / 3.0)
    return feff if MIN_FEFF_L <= feff <= MAX_FEFF_L else MIN_FEFF_L


def _darmenova_drag_partition_vectorized(rdrag: NDArray, vegfrac: NDArray) -> NDArray:
    """
    Vectorized implementation of the Darmenova drag partition scheme.

    Parameters
    ----------
    rdrag : NDArray
        1D array of the drag partition parameter for valid cells.
    vegfrac : NDArray
        1D array of the vegetation fraction for valid cells.

    Returns
    -------
    NDArray
        1D array of the calculated drag partition factor.
    """
    # Vectorized darmenova_drag_partition logic
    feff_veg = np.full_like(vegfrac, DRAG_MIN_VAL)
    mask_veg = (vegfrac >= 0.0) & (vegfrac < VEG_THRESHOLD_FENGSHA)
    if np.any(mask_veg):
        # Use np.errstate to avoid log(0) warnings for values outside the mask
        with np.errstate(divide="ignore"):
            Lc_veg = -0.35 * np.log(1.0 - vegfrac[mask_veg])
        R1 = 1.0 / np.sqrt(1.0 - SIGV_D * MV_D * Lc_veg)
        R2 = 1.0 / np.sqrt(1.0 + MV_D * BETAV_D * Lc_veg)
        feff_veg[mask_veg] = R1 * R2

    # Use np.errstate to avoid divide-by-zero warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        Lc_bare = rdrag / (1.0 - vegfrac)
    tmpVal = 1.0 - SIGB_D * MB_D * Lc_bare
    feff_bare = np.full_like(rdrag, DRAG_MIN_VAL)
    mask_bare = ~(
        (vegfrac < 0.0)
        | (vegfrac >= VEG_THRESHOLD_FENGSHA)
        | (rdrag > 0.2)
        | (tmpVal <= 0.0)
    )
    if np.any(mask_bare):
        R1_b = 1.0 / np.sqrt(1.0 - SIGB_D * MB_D * Lc_bare[mask_bare])
        R2_b = 1.0 / np.sqrt(1.0 + MB_D * BETAB_D * Lc_bare[mask_bare])
        feff_bare[mask_bare] = R1_b * R2_b

    feff = feff_veg * feff_bare
    return np.where((feff >= 1.0e-5) & (feff <= 1.0), feff, DRAG_MIN_VAL)


def _leung_drag_partition_vectorized(
    rdrag: NDArray, vegfrac: NDArray, lai: NDArray
) -> NDArray:
    """
    Vectorized implementation of the Leung drag partition scheme.

    Parameters
    ----------
    rdrag : NDArray
        1D array of the drag partition parameter for valid cells.
    vegfrac : NDArray
        1D array of the vegetation fraction for valid cells.
    lai : NDArray
        1D array of the Leaf Area Index for valid cells.

    Returns
    -------
    NDArray
        1D array of the calculated drag partition factor.
    """
    # Vectorized leung_drag_partition logic
    SMALL_VAL = 1.0e-10
    frac_bare = np.clip(1.0 - lai / VEG_THRESHOLD_FENGSHA, SMALL_VAL, 1.0)

    feff_veg = np.zeros_like(lai)
    mask_veg = (lai > 0.0) & (lai < VEG_THRESHOLD_FENGSHA)
    if np.any(mask_veg):
        K = 2.0 * (1.0 / np.maximum(1.0 - lai[mask_veg], SMALL_VAL) - 1.0)
        feff_veg[mask_veg] = (K + F0_L * C_L) / (K + C_L)

    feff_bare = np.zeros_like(rdrag)
    mask_bare = (rdrag > 0.0) & (rdrag <= 0.2) & (lai < VEG_THRESHOLD_FENGSHA)
    if np.any(mask_bare):
        Lc_bare = rdrag[mask_bare] / np.maximum(frac_bare[mask_bare], SMALL_VAL)
        tmpVal = 1.0 - SIGB_L * MB_L * Lc_bare

        sub_feff_bare = np.zeros_like(Lc_bare)
        mask_tmp = tmpVal > SMALL_VAL
        if np.any(mask_tmp):
            Rbare1 = 1.0 / np.sqrt(
                np.maximum(1.0 - SIGB_L * MB_L * Lc_bare[mask_tmp], SMALL_VAL)
            )
            Rbare2 = 1.0 / np.sqrt(1.0 + BETAB_L * MB_L * Lc_bare[mask_tmp])
            sub_feff_bare[mask_tmp] = Rbare1 * Rbare2
        feff_bare[mask_bare] = sub_feff_bare

    feff = (vegfrac * feff_veg**3 + frac_bare * feff_bare**3) ** (1.0 / 3.0)
    return np.where((feff >= MIN_FEFF_L) & (feff <= MAX_FEFF_L), feff, MIN_FEFF_L)


def _calculate_drag_partition(
    rdrag: NDArray, vegfrac: NDArray, lai: NDArray, drag_opt: int
) -> NDArray:
    """
    Calculates the drag partition factor (R) based on the selected scheme.

    This function dispatches to the appropriate vectorized drag partition
    scheme based on the `drag_opt` parameter.

    Parameters
    ----------
    rdrag : NDArray
        1D array of the drag partition parameter for valid cells.
    vegfrac : NDArray
        1D array of the vegetation fraction for valid cells.
    lai : NDArray
        1D array of the Leaf Area Index for valid cells.
    drag_opt : int
        Drag option (1, 2, or 3).

    Returns
    -------
    NDArray
        1D array of the calculated drag partition factor (R) for valid cells.
    """
    if drag_opt == 2:
        return _darmenova_drag_partition_vectorized(rdrag, vegfrac)
    elif drag_opt == 3:
        return _leung_drag_partition_vectorized(rdrag, vegfrac, lai)

    # Default case for drag_opt == 1 or any other value
    return rdrag


def dust_emission_fengsha(
    fraclake: NDArray,
    fracsnow: NDArray,
    oro: NDArray,
    slc: NDArray,
    clay: NDArray,
    sand: NDArray,
    ssm: NDArray,
    rdrag: NDArray,
    airdens: NDArray,
    ustar: NDArray,
    vegfrac: NDArray,
    lai: NDArray,
    uthrs: NDArray,
    alpha: float,
    gamma: float,
    kvhmax: float,
    grav: float,
    distribution: NDArray,
    drylimit_factor: float,
    moist_correct: float,
    drag_opt: int,
) -> NDArray:
    """
    Compute dust emissions using NOAA/ARL FENGSHA model (Vectorized).

    This version is fully vectorized using NumPy and does not contain explicit
    loops over spatial dimensions, providing a significant performance
    improvement for large arrays.

    Parameters
    ----------
    fraclake : NDArray
        2D array of the fraction of lake coverage.
    fracsnow : NDArray
        2D array of the fraction of snow coverage.
    oro : NDArray
        2D array of the land/water mask (1 for land).
    slc : NDArray
        2D array of the soil liquid content.
    clay : NDArray
        2D array of the clay fraction.
    sand : NDArray
        2D array of the sand fraction.
    ssm : NDArray
        2D array of the surface soil moisture.
    rdrag : NDArray
        2D array of the drag partition parameter.
    airdens : NDArray
        2D array of the air density.
    ustar : NDArray
        2D array of the friction velocity.
    vegfrac : NDArray
        2D array of the vegetation fraction.
    lai : NDArray
        2D array of the Leaf Area Index.
    uthrs : NDArray
        2D array of the threshold velocity.
    alpha : float
        Tuning parameter.
    gamma : float
        Tuning parameter.
    kvhmax : float
        Max KVH ratio.
    grav : float
        Gravity acceleration.
    distribution : NDArray
        1D array of the size distribution per bin.
    drylimit_factor : float
        Dry limit factor for moisture correction.
    moist_correct : float
        Moisture correction factor.
    drag_opt : int
        Drag option (1, 2, or 3).

    Returns
    -------
    NDArray
        3D array of emissions of shape (ni, nj, nbins).
    """
    # --- Create a mask for valid grid cells to perform calculations on ---
    valid_mask = (
        (oro == LAND_MASK_VALUE)
        & (ssm >= SSM_THRESHOLD)
        & (clay >= 0.0)
        & (sand >= 0.0)
    )

    if drag_opt == 2:
        valid_mask &= ~(
            (vegfrac < 0.0)
            | (vegfrac >= VEG_THRESHOLD_FENGSHA)
            | (rdrag > MAX_RDRAG_FENGSHA)
        )
    elif drag_opt == 3:
        valid_mask &= ~((vegfrac < 0.0) | (lai >= VEG_THRESHOLD_FENGSHA))
    else:  # drag_opt == 1 or other default case
        valid_mask &= rdrag >= 0.0

    # Initialize emissions array to zeros
    emissions = np.zeros(fraclake.shape + (len(distribution),), dtype=np.float64)

    # If no cells are valid, we can return early
    if not np.any(valid_mask):
        return emissions

    # --- Perform calculations only on the valid cells ---
    # Slicing with the mask flattens the array, which is fine as we'll place it back later.
    fracland = np.maximum(0.0, 1.0 - fraclake[valid_mask]) * np.maximum(
        0.0, 1.0 - fracsnow[valid_mask]
    )

    # Vectorized mb95_vertical_flux_ratio
    clay_v = clay[valid_mask]
    kvh = np.full_like(clay_v, kvhmax)
    sub_mask = clay_v <= 0.2
    kvh[sub_mask] = 10.0 ** (13.4 * clay_v[sub_mask] - 6.0)

    alpha_grav = alpha / max(grav, 1.0e-10)
    total_emissions = (
        alpha_grav * fracland * (ssm[valid_mask] ** gamma) * airdens[valid_mask] * kvh
    )

    # --- Drag Partition Calculation (Vectorized) ---
    rdrag_v = rdrag[valid_mask]
    vegfrac_v = vegfrac[valid_mask]
    lai_v = lai[valid_mask]
    R = _calculate_drag_partition(rdrag_v, vegfrac_v, lai_v, drag_opt)
    rustar = R * ustar[valid_mask]

    # --- Moisture Correction (Vectorized) ---
    smois = slc[valid_mask] * moist_correct
    sand_v = sand[valid_mask]
    # Vectorized gocart_vol_to_grav
    vsat = 0.489 - 0.126 * sand_v
    gravimetric_soil_moisture = (
        smois
        * WATER_DENSITY_GCM3
        * 1000.0
        / (GOCART_PARTICLE_DENSITY_GCM3 * 1000.0 * (1.0 - vsat))
        * 100.0
    )
    fecan_dry_limit_val = (
        drylimit_factor * clay_v * (FECAN_CLAY_COEFF_A * clay_v + FECAN_CLAY_COEFF_B)
    )

    correction_term = np.maximum(0.0, gravimetric_soil_moisture - fecan_dry_limit_val)
    h = np.sqrt(1.0 + 1.21 * correction_term**0.68)

    u_thresh = uthrs[valid_mask] * h

    # --- Horizontal Flux Calculation (Vectorized) ---
    u_sum = rustar + u_thresh
    q = np.maximum(0.0, rustar - u_thresh) * u_sum * u_sum

    # --- Final Emission Calculation and Broadcasting ---
    # Reshape for broadcasting: (n_valid,) -> (n_valid, 1)
    # distribution has shape (nbins,) -> (1, nbins)
    # Resulting shape after broadcasting: (n_valid, nbins)
    final_emissions_v = (total_emissions * q)[:, np.newaxis] * distribution[
        np.newaxis, :
    ]

    # Place the calculated values back into the full-sized emissions array
    emissions[valid_mask] = final_emissions_v

    return emissions


def dust_emission_gocart2g(
    radius: NDArray,
    fraclake: NDArray,
    gwettop: NDArray,
    oro: NDArray,
    u10m: NDArray,
    v10m: NDArray,
    Ch_DU: float,
    du_src: NDArray,
    grav: float,
) -> NDArray:
    """
    Computes dust emissions using GOCART2G scheme (Vectorized).

    This version is fully vectorized using NumPy and does not contain explicit
    loops over spatial dimensions, providing a significant performance
    improvement for large arrays.

    Parameters
    ----------
    radius: NDArray
        1D array of particle radii (nbins,).
    fraclake: NDArray
        2D array of the fraction of lake coverage (ni, nj).
    gwettop: NDArray
        2D array of surface wetness (ni, nj).
    oro: NDArray
        2D array of the land/water mask (1 for land) (ni, nj).
    u10m: NDArray
        2D array of the 10m u-wind component (ni, nj).
    v10m: NDArray
        2D array of the 10m v-wind component (ni, nj).
    Ch_DU: float
        Dust emission coefficient.
    du_src: NDArray
        2D array of the dust source function (ni, nj).
    grav: float
        Gravity acceleration.

    Returns
    -------
    NDArray
        3D array of emissions of shape (ni, nj, nbins).
    """
    # --- Pre-calculations and constants ---
    air_dens = 1.25
    soil_density = SOIL_DENSITY_GCM3 * 1000.0  # to kg/m3
    ni, nj = u10m.shape
    nbins = len(radius)

    # --- Vectorized threshold velocity calculation ---
    # `diameter` has shape (nbins,)
    diameter = 2.0 * radius

    # `u_thresh0` has shape (nbins,)
    term1 = np.sqrt(soil_density * grav * diameter / air_dens)
    term2 = np.sqrt(1.0 + 6.0e-7 / (soil_density * grav * diameter**2.5))
    term3 = np.sqrt(1.928 * (1331.0 * (100.0 * diameter) ** 1.56 + 0.38) ** 0.092 - 1.0)
    u_thresh0 = 0.13 * term1 * term2 / term3

    # --- Vectorized wind speed ---
    # `w10m` has shape (ni, nj)
    w10m = np.sqrt(u10m**2 + v10m**2)

    # --- Create a mask for valid grid cells ---
    # All masks have shape (ni, nj)
    valid_mask = (oro == LAND_MASK_VALUE) & (gwettop < 0.5)

    # Initialize emissions array to zeros
    emissions = np.zeros((ni, nj, nbins), dtype=np.float64)

    # If no cells are valid, we can return early
    if not np.any(valid_mask):
        return emissions

    # --- Perform calculations only on the valid cells ---
    # Slice 2D arrays to get flattened 1D arrays of valid cells
    gwettop_v = gwettop[valid_mask]
    w10m_v = w10m[valid_mask]
    fraclake_v = fraclake[valid_mask]
    du_src_v = du_src[valid_mask]

    # --- Vectorized threshold calculation for valid cells ---
    # Shape broadcasting: `gwettop_v` (n_valid,) -> (n_valid, 1)
    # `u_thresh0` (nbins,) -> (1, nbins)
    # `u_thresh` has shape (n_valid, nbins)
    log_gwettop = np.log10(np.maximum(1.0e-3, gwettop_v))
    u_thresh = u_thresh0[np.newaxis, :] * (1.2 + 0.2 * log_gwettop[:, np.newaxis])
    u_thresh = np.maximum(0.0, u_thresh)

    # --- Vectorized emission calculation ---
    # Shape broadcasting: `w10m_v` (n_valid,) -> (n_valid, 1)
    # `w10m_v` is compared against each bin's threshold in `u_thresh`
    # `flux` has shape (n_valid, nbins)
    flux = (w10m_v[:, np.newaxis] ** 2) * (w10m_v[:, np.newaxis] - u_thresh)

    # Apply wind threshold mask (where w10m > u_thresh)
    flux = np.where(w10m_v[:, np.newaxis] > u_thresh, flux, 0.0)

    # Apply lake and source function scaling
    # Broadcasting: `fraclake_v` and `du_src_v` (n_valid,) -> (n_valid, 1)
    scaled_flux = (
        Ch_DU * (1.0 - fraclake_v[:, np.newaxis]) * du_src_v[:, np.newaxis] * flux
    )

    # Place the calculated values back into the full-sized emissions array
    emissions[valid_mask] = scaled_flux

    return emissions
