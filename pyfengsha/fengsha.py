"""
NOAA/ARL FENGSHA dust emission model and GOCART2G scheme implementations.
"""

import math
import numpy as np
from numba import jit

# --- Constants ---
# Using descriptive names for physical and model constants.
# No more magic numbers.

# Physical Constants
G_ACCEL_CMS2 = 9.81 * 100.0  # Gravity in cm/s^2

# Soil and Water Properties
SOIL_DENSITY_GCM3 = 2.65
WATER_DENSITY_GCM3 = 1.0
GOCART_PARTICLE_DENSITY_GCM3 = 1.7

# Fecan Moisture Correction Parameters
FECAN_CLAY_COEFF_A = 14.0
FECAN_CLAY_COEFF_B = 17.0
FECAN_MOISTURE_COEFF_A = 1.21
FECAN_MOISTURE_COEFF_B = 0.68

# Drag Partition Scheme Constants (Darmenova)
DRAG_MIN_VAL = 1.0e-3
SIGB_D = 1.0
MB_D = 0.5
BETAB_D = 90.0
SIGV_D = 1.45
MV_D = 0.16
BETAV_D = 202.0

# Drag Partition Scheme Constants (Leung)
LAI_THRESHOLD_L = 0.33
C_L = 4.8
F0_L = 0.32
SIGB_L = 1.0
MB_L = 0.5
BETAB_L = 90.0
MIN_FEFF_L = 1.0E-5
MAX_FEFF_L = 1.0

# Thresholds and masks for main emission calculation
SSM_THRESHOLD = 1.0E-02
VEG_THRESHOLD_FENGSHA = 0.4
LAND_MASK_VALUE = 1.0
MAX_RDRAG_FENGSHA = 0.3

# --- Helper Functions ---

@jit(nopython=True)
def volumetric_to_gravimetric(vsoil: float, sandfrac: float) -> float:
    """
    Convert volumetric soil moisture to gravimetric based on FENGSHA scheme.

    Args:
        vsoil: Volumetric Soil Moisture [m3/m3].
        sandfrac: Fractional Sand content [0-1].

    Returns:
        Gravimetric soil moisture [kg/kg].
    """
    vsat = 0.489 - 0.00126 * (sandfrac * 100.0)
    return vsoil * WATER_DENSITY_GCM3 * 1000.0 / (SOIL_DENSITY_GCM3 * 1000.0 * (1.0 - vsat))

@jit(nopython=True)
def gocart_vol_to_grav(vsoil: float, sandfrac: float) -> float:
    """
    Convert volumetric soil moisture to gravimetric based on GOCART scheme.
    Note: The original returned a value scaled by 100. This is now done
    at the call site for clarity.

    Args:
        vsoil: Volumetric Soil Moisture [m3/m3].
        sandfrac: Fractional Sand content [0-1].

    Returns:
        Gravimetric soil moisture [kg/kg].
    """
    vsat = 0.489 - 0.126 * sandfrac
    return vsoil * WATER_DENSITY_GCM3 * 1000.0 / (GOCART_PARTICLE_DENSITY_GCM3 * 1000.0 * (1.0 - vsat))

@jit(nopython=True)
def fecan_dry_limit(clay: float) -> float:
    """Calculates the Fecan dry limit for soil moisture."""
    if clay <= 0.0:
        # Using an epsilon for stability with zero clay
        clay = 1.0e-4
    return FECAN_CLAY_COEFF_A * clay * clay + FECAN_CLAY_COEFF_B * clay

@jit(nopython=True)
def fecan_moisture_correction(vol_soil_moisture: float, sand: float, clay: float) -> float:
    """
    Calculates the Fecan soil moisture correction factor (H).
    """
    gravsm = volumetric_to_gravimetric(vol_soil_moisture, sand)
    drylimit = fecan_dry_limit(clay)

    if gravsm > drylimit:
        return np.sqrt(1.0 + FECAN_MOISTURE_COEFF_A * (gravsm - drylimit)**FECAN_MOISTURE_COEFF_B)
    return 1.0

@jit(nopython=True)
def gocart_moisture_correction(slc: float, sand: float, clay: float, b: float) -> float:
    """
    GOCART version of moisture correction.
    """
    # Note: GOCART scales gravimetric moisture by 100.
    gravimetric_soil_moisture = gocart_vol_to_grav(slc, sand) * 100.0
    fecan_dry_limit_val = b * clay * (FECAN_CLAY_COEFF_A * clay + FECAN_CLAY_COEFF_B)

    return np.sqrt(1.0 + 1.21 * max(0.0, gravimetric_soil_moisture - fecan_dry_limit_val)**0.68)

@jit(nopython=True)
def shao_1996_soil_moisture(w: float) -> float:
    """Shao 1996 soil moisture function."""
    return np.exp(22.7 * w)

@jit(nopython=True)
def shao_2004_soil_moisture(w: float) -> float:
    """Shao 2004 soil moisture function."""
    if w <= 0.03:
        return np.exp(22.7 * w)
    return np.exp(95.3 * w - 2.029)

@jit(nopython=True)
def modified_threshold_velocity(u_ts0: float, H: float, drag: float) -> float:
    """Calculate modified threshold velocity."""
    return u_ts0 * H / drag

@jit(nopython=True)
def mb95_vertical_flux_ratio(clay: float, max_ratio: float = 2.0E-4) -> float:
    """
    Marticorena and Bergametti (1995) vertical to horizontal flux ratio.
    """
    if clay <= 0.2:
        return 10.0**(13.4 * clay - 6.0)
    return max_ratio

@jit(nopython=True)
def horizontal_saltation_flux(ust: float, utst: float) -> float:
    """Calculates Horizontal Saltation Flux Q."""
    if ust <= utst:
        return 0.0
    return ust * (ust * ust - utst * utst)

@jit(nopython=True)
def mackinnon_drag_partition(z0: float) -> float:
    """MacKinnon drag partition."""
    z0s = 1.0e-04
    return 1.0 - np.log(z0 / z0s) / np.log(0.7 * (12255.0 / z0s) ** 0.8)

@jit(nopython=True)
def mb95_drag_partition(z0: float) -> float:
    """Marticorena and Bergametti (1995) drag partition."""
    z0s = 1.0e-04
    return 1.0 - np.log(z0 / z0s) / np.log(0.7 * (10.0 / z0s) ** 0.8)

@jit(nopython=True)
def kok_aerosol_distribution(radius: np.ndarray, r_low: np.ndarray, r_up: np.ndarray) -> np.ndarray:
    """
    Compute Kok's dust size aerosol distribution.
    """
    median_mass_diameter = 3.4
    geom_std_dev = 3.0
    crack_prop_len = 12.0
    factor = 1.0 / (np.sqrt(2.0) * np.log(geom_std_dev))

    num_bins = len(radius)
    distribution = np.zeros(num_bins)
    total_volume = 0.0

    for n in range(num_bins):
        diameter = 2.0 * radius[n]
        dlam = diameter / crack_prop_len
        dist_val = diameter * (1.0 + math.erf(factor * np.log(diameter / median_mass_diameter))) * \
                   np.exp(-dlam**3) * np.log(r_up[n] / r_low[n])
        distribution[n] = dist_val
        total_volume += dist_val

    return distribution / total_volume

# --- Main Emission Schemes ---

@jit(nopython=True)
def fengsha(rho_phy: float, smois: float, ssm: float, xland: float, ust: float, clay: float, sand: float, rdrag: float, u_ts0: float) -> float:
    """
    Core Fengsha dust emission calculation.
    """
    if xland != LAND_MASK_VALUE or ssm <= 0:
        return 0.0

    H = fecan_moisture_correction(smois, sand, clay)
    kvh = mb95_vertical_flux_ratio(clay)
    u_ts = modified_threshold_velocity(u_ts0, H, rdrag)
    Q = horizontal_saltation_flux(ust, u_ts)

    return ssm * rho_phy / G_ACCEL_CMS2 * kvh * Q

@jit(nopython=True)
def fengsha_albedo(rho_phy: float, smois: float, ssm: float, xland: float, ust: float, clay: float, sand: float, rdrag: float, u_ts0: float) -> float:
    """
    Calculate dust emission using Fengsha albedo based logic.
    """
    if xland != LAND_MASK_VALUE or ssm <= 0:
        return 0.0

    H = fecan_moisture_correction(smois, sand, clay)
    kvh = mb95_vertical_flux_ratio(clay)
    u_ts = modified_threshold_velocity(u_ts0, H, rdrag)

    ustar_albedo = ust * rdrag
    Q = horizontal_saltation_flux(ustar_albedo, u_ts)

    return ssm * rho_phy / G_ACCEL_CMS2 * kvh * Q


def darmenova_drag_partition_v(Lc: np.ndarray, vegfrac: np.ndarray, thresh: float) -> np.ndarray:
    """
    Vectorized Darmenova drag partition scheme.

    Parameters
    ----------
    Lc : np.ndarray
        Roughness length.
    vegfrac : np.ndarray
        Vegetation fraction.
    thresh : float
        Vegetation threshold.

    Returns
    -------
    np.ndarray
        The effective fraction of partitioned drag.
    """
    # Vegetated fraction effect
    feff_veg = np.full_like(vegfrac, DRAG_MIN_VAL)
    mask_veg = (vegfrac >= 0.0) & (vegfrac < thresh)
    with np.errstate(divide='ignore', invalid='ignore'):
        Lc_veg = -0.35 * np.log(1.0 - vegfrac[mask_veg])
    R1 = 1.0 / np.sqrt(1.0 - SIGV_D * MV_D * Lc_veg)
    R2 = 1.0 / np.sqrt(1.0 + MV_D * BETAV_D * Lc_veg)
    feff_veg[mask_veg] = R1 * R2

    # Bare soil fraction effect
    with np.errstate(divide='ignore', invalid='ignore'):
        Lc_bare = Lc / (1.0 - vegfrac)
    tmpVal = 1.0 - SIGB_D * MB_D * Lc_bare
    feff_bare = np.full_like(Lc, DRAG_MIN_VAL)
    mask_bare = ~((vegfrac < 0.0) | (vegfrac >= thresh) | (Lc > 0.2) | (tmpVal <= 0.0))
    R1_b = 1.0 / np.sqrt(1.0 - SIGB_D * MB_D * Lc_bare[mask_bare])
    R2_b = 1.0 / np.sqrt(1.0 + MB_D * BETAB_D * Lc_bare[mask_bare])
    feff_bare[mask_bare] = R1_b * R2_b

    feff = feff_veg * feff_bare
    return np.where((feff >= 1.0e-5) & (feff <= 1.0), feff, DRAG_MIN_VAL)


def leung_drag_partition_v(Lc: np.ndarray, lai: np.ndarray, gvf: np.ndarray, thresh: float) -> np.ndarray:
    """
    Vectorized Leung drag partition scheme.

    Parameters
    ----------
    Lc : np.ndarray
        Roughness length.
    lai : np.ndarray
        Leaf Area Index.
    gvf : np.ndarray
        Green vegetation fraction.
    thresh : float
        LAI threshold.

    Returns
    -------
    np.ndarray
        The effective fraction of partitioned drag.
    """
    SMALL_VAL = 1.0E-10
    with np.errstate(divide='ignore', invalid='ignore'):
        frac_bare = np.clip(1.0 - lai / thresh, SMALL_VAL, 1.0)

    feff_veg = np.zeros_like(lai)
    mask_veg = (lai > 0.0) & (lai < thresh)
    K = 2.0 * (1.0 / np.maximum(1.0 - lai[mask_veg], SMALL_VAL) - 1.0)
    feff_veg[mask_veg] = (K + F0_L * C_L) / (K + C_L)

    feff_bare = np.zeros_like(Lc)
    mask_bare = (Lc > 0.0) & (Lc <= 0.2) & (lai < thresh)
    Lc_bare_masked = Lc[mask_bare] / np.maximum(frac_bare[mask_bare], SMALL_VAL)
    tmpVal = 1.0 - SIGB_L * MB_L * Lc_bare_masked
    sub_feff_bare = np.zeros_like(Lc_bare_masked)
    mask_tmp = tmpVal > SMALL_VAL
    Rbare1 = 1.0 / np.sqrt(np.maximum(1.0 - SIGB_L * MB_L * Lc_bare_masked[mask_tmp], SMALL_VAL))
    Rbare2 = 1.0 / np.sqrt(1.0 + BETAB_L * MB_L * Lc_bare_masked[mask_tmp])
    sub_feff_bare[mask_tmp] = Rbare1 * Rbare2
    feff_bare[mask_bare] = sub_feff_bare

    feff = (gvf * feff_veg**3 + frac_bare * feff_bare**3) ** (1.0/3.0)
    return np.where((feff >= MIN_FEFF_L) & (feff <= MAX_FEFF_L), feff, MIN_FEFF_L)


def _calculate_kvh_v(clay: np.ndarray, kvhmax: float) -> np.ndarray:
    """Vectorized calculation of the vertical flux ratio."""
    kvh = np.full_like(clay, kvhmax)
    sub_mask = clay <= 0.2
    kvh[sub_mask] = 10.0**(13.4 * clay[sub_mask] - 6.0)
    return kvh


def _calculate_moisture_correction_v(slc: np.ndarray, sand: np.ndarray, clay: np.ndarray,
                                     drylimit_factor: float, moist_correct: float) -> np.ndarray:
    """Vectorized calculation of the GOCART moisture correction factor (h)."""
    smois = slc * moist_correct
    vsat = 0.489 - 0.126 * sand
    gravimetric_soil_moisture = smois * WATER_DENSITY_GCM3 * 1000.0 / \
        (GOCART_PARTICLE_DENSITY_GCM3 * 1000.0 * (1.0 - vsat)) * 100.0
    fecan_dry_limit_val = drylimit_factor * clay * (FECAN_CLAY_COEFF_A * clay + FECAN_CLAY_COEFF_B)
    correction_term = np.maximum(0.0, gravimetric_soil_moisture - fecan_dry_limit_val)
    return np.sqrt(1.0 + 1.21 * correction_term**0.68)


def _calculate_horizontal_flux_v(ustar: np.ndarray, uthrs: np.ndarray, R: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Vectorized calculation of the horizontal saltation flux (q)."""
    rustar = R * ustar
    u_thresh = uthrs * h
    u_sum = rustar + u_thresh
    return np.maximum(0.0, rustar - u_thresh) * u_sum * u_sum


def dust_emission_fengsha(
    fraclake: np.ndarray, fracsnow: np.ndarray, oro: np.ndarray, slc: np.ndarray,
    clay: np.ndarray, sand: np.ndarray, ssm: np.ndarray, rdrag: np.ndarray,
    airdens: np.ndarray, ustar: np.ndarray, vegfrac: np.ndarray, lai: np.ndarray,
    uthrs: np.ndarray, alpha: float, gamma: float, kvhmax: float, grav: float,
    distribution: np.ndarray, drylimit_factor: float, moist_correct: float, drag_opt: int
) -> np.ndarray:
    """
    Compute dust emissions using NOAA/ARL FENGSHA model (Vectorized).

    This version is fully vectorized using NumPy and does not contain explicit
    loops over spatial dimensions, providing a significant performance
    improvement for large arrays.

    Parameters
    ----------
    fraclake : np.ndarray
        2D array of the fraction of lake coverage.
    fracsnow : np.ndarray
        2D array of the fraction of snow coverage.
    oro : np.ndarray
        2D array of the land/water mask (1 for land).
    slc : np.ndarray
        2D array of the soil liquid content.
    clay : np.ndarray
        2D array of the clay fraction.
    sand : np.ndarray
        2D array of the sand fraction.
    ssm : np.ndarray
        2D array of the surface soil moisture.
    rdrag : np.ndarray
        2D array of the drag partition parameter.
    airdens : np.ndarray
        2D array of the air density.
    ustar : np.ndarray
        2D array of the friction velocity.
    vegfrac : np.ndarray
        2D array of the vegetation fraction.
    lai : np.ndarray
        2D array of the Leaf Area Index.
    uthrs : np.ndarray
        2D array of the threshold velocity.
    alpha : float
        Tuning parameter.
    gamma : float
        Tuning parameter.
    kvhmax : float
        Max KVH ratio.
    grav : float
        Gravity acceleration.
    distribution : np.ndarray
        1D array of the size distribution per bin.
    drylimit_factor : float
        Dry limit factor for moisture correction.
    moist_correct : float
        Moisture correction factor.
    drag_opt : int
        Drag option (1, 2, or 3).

    Returns
    -------
    np.ndarray
        3D array of emissions of shape (ni, nj, nbins).
    """
    # --- Create a mask for valid grid cells to perform calculations on ---
    valid_mask = (oro == LAND_MASK_VALUE) & \
                 (ssm >= SSM_THRESHOLD) & \
                 (clay >= 0.0) & \
                 (sand >= 0.0)

    if drag_opt == 2:
        valid_mask &= ~((vegfrac < 0.0) | (vegfrac >= VEG_THRESHOLD_FENGSHA) | (rdrag > MAX_RDRAG_FENGSHA))
    elif drag_opt == 3:
        valid_mask &= ~((vegfrac < 0.0) | (lai >= VEG_THRESHOLD_FENGSHA))
    else:  # drag_opt == 1 or other default case
        valid_mask &= (rdrag >= 0.0)

    # Initialize emissions array to zeros
    emissions = np.zeros(fraclake.shape + (len(distribution),), dtype=np.float64)

    # If no cells are valid, we can return early
    if not np.any(valid_mask):
        return emissions

    # --- Orchestration of Calculations ---
    # All calculations are performed only on the valid grid cells.
    fracland = np.maximum(0.0, 1.0 - fraclake[valid_mask]) * \
               np.maximum(0.0, 1.0 - fracsnow[valid_mask])

    kvh = _calculate_kvh_v(clay[valid_mask], kvhmax)

    alpha_grav = alpha / max(grav, 1.0E-10)
    total_emissions = alpha_grav * fracland * (ssm[valid_mask] ** gamma) * airdens[valid_mask] * kvh

    if drag_opt == 1:
        R = rdrag[valid_mask]
    elif drag_opt == 2:
        R = darmenova_drag_partition_v(rdrag[valid_mask], vegfrac[valid_mask], VEG_THRESHOLD_FENGSHA)
    elif drag_opt == 3:
        R = leung_drag_partition_v(rdrag[valid_mask], lai[valid_mask], vegfrac[valid_mask], VEG_THRESHOLD_FENGSHA)
    else:
        R = rdrag[valid_mask]

    h = _calculate_moisture_correction_v(slc[valid_mask], sand[valid_mask], clay[valid_mask],
                                         drylimit_factor, moist_correct)

    q = _calculate_horizontal_flux_v(ustar[valid_mask], uthrs[valid_mask], R, h)

    # --- Final Emission Calculation and Broadcasting ---
    # Reshape for broadcasting: (n_valid,) -> (n_valid, 1)
    # distribution has shape (nbins,) -> (1, nbins)
    # Resulting shape after broadcasting: (n_valid, nbins)
    final_emissions_v = (total_emissions * q)[:, np.newaxis] * distribution[np.newaxis, :]

    # Place the calculated values back into the full-sized emissions array
    emissions[valid_mask] = final_emissions_v

    return emissions

@jit(nopython=True)
def dust_emission_gocart2g(
    radius: np.ndarray, fraclake: np.ndarray, gwettop: np.ndarray, oro: np.ndarray,
    u10m: np.ndarray, v10m: np.ndarray, Ch_DU: float, du_src: np.ndarray, grav: float
) -> np.ndarray:
    """
    Computes dust emissions using GOCART2G scheme.

    Args:
        radius: Particle radii.
        fraclake: Fraction of lake coverage.
        gwettop: Surface wetness.
        oro: Land mask.
        u10m: 10m u-wind component.
        v10m: 10m v-wind component.
        Ch_DU: Dust emission coefficient.
        du_src: Dust source function.
        grav: Gravity.

    Returns:
        Emissions array.
    """
    air_dens = 1.25
    soil_density = SOIL_DENSITY_GCM3 * 1000.0  # to kg/m3

    nbins = len(radius)
    ni, nj = u10m.shape
    emissions = np.zeros((ni, nj, nbins))

    for n in range(nbins):
        diameter = 2.0 * radius[n]
        u_thresh0 = 0.13 * np.sqrt(soil_density * grav * diameter / air_dens) * \
                    np.sqrt(1.0 + 6.0e-7 / (soil_density * grav * diameter**2.5)) / \
                    np.sqrt(1.928 * (1331.0 * (100.0 * diameter)**1.56 + 0.38)**0.092 - 1.0)

        for j in range(nj):
            for i in range(ni):
                if oro[i, j] != LAND_MASK_VALUE:
                    continue

                w10m = np.sqrt(u10m[i, j]**2 + v10m[i, j]**2)
                if gwettop[i, j] < 0.5:
                    u_thresh = max(0.0, u_thresh0 * (1.2 + 0.2 * np.log10(max(1.e-3, gwettop[i, j]))))
                    if w10m > u_thresh:
                        emissions[i, j, n] = (1.0 - fraclake[i, j]) * w10m**2 * (w10m - u_thresh)

    # Apply scaling and source function
    for j in range(nj):
        for i in range(ni):
            emissions[i, j, :] *= Ch_DU * du_src[i, j]

    return emissions
