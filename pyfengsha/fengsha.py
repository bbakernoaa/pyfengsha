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

@jit(nopython=True)
def darmenova_drag_partition(Lc: float, vegfrac: float, thresh: float) -> float:
    """Darmenova drag partition scheme."""
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
    """Leung drag partition scheme."""
    SMALL_VAL = 1.0E-10
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

    feff = (gvf * feff_veg**3 + frac_bare * feff_bare**3) ** (1.0/3.0)
    return feff if MIN_FEFF_L <= feff <= MAX_FEFF_L else MIN_FEFF_L

@jit(nopython=True)
def dust_emission_fengsha(
    fraclake: np.ndarray, fracsnow: np.ndarray, oro: np.ndarray, slc: np.ndarray,
    clay: np.ndarray, sand: np.ndarray, ssm: np.ndarray, rdrag: np.ndarray,
    airdens: np.ndarray, ustar: np.ndarray, vegfrac: np.ndarray, lai: np.ndarray,
    uthrs: np.ndarray, alpha: float, gamma: float, kvhmax: float, grav: float,
    distribution: np.ndarray, drylimit_factor: float, moist_correct: float, drag_opt: int
) -> np.ndarray:
    """
    Compute dust emissions using NOAA/ARL FENGSHA model.

    Args:
        fraclake: Fraction of lake coverage.
        fracsnow: Fraction of snow coverage.
        oro: Land/water mask (1 for land).
        slc: Soil liquid content.
        clay: Clay fraction.
        sand: Sand fraction.
        ssm: Surface soil moisture.
        rdrag: Drag partition parameter.
        airdens: Air density.
        ustar: Friction velocity.
        vegfrac: Vegetation fraction.
        lai: Leaf Area Index.
        uthrs: Threshold velocity.
        alpha: Tuning parameter.
        gamma: Tuning parameter.
        kvhmax: Max KVH ratio.
        grav: Gravity acceleration.
        distribution: Size distribution per bin.
        drylimit_factor: Dry limit factor for moisture correction.
        moist_correct: Moisture correction factor.
        drag_opt: Drag option (1, 2, or 3).

    Returns:
        Emissions array of shape (ni, nj, nbins).
    """
    ni, nj = fraclake.shape
    nbins = len(distribution)
    emissions = np.zeros((ni, nj, nbins))
    alpha_grav = alpha / max(grav, 1.0E-10)

    for j in range(nj):
        for i in range(ni):
            # --- Pre-computation checks ---
            if (oro[i, j] != LAND_MASK_VALUE or
                (drag_opt == 2 and (vegfrac[i, j] < 0.0 or vegfrac[i, j] >= VEG_THRESHOLD_FENGSHA or rdrag[i, j] > MAX_RDRAG_FENGSHA)) or
                (drag_opt == 3 and (vegfrac[i, j] < 0.0 or lai[i, j] >= VEG_THRESHOLD_FENGSHA)) or
                (drag_opt not in [2, 3] and rdrag[i, j] < 0.0) or
                ssm[i, j] < SSM_THRESHOLD or clay[i, j] < 0.0 or sand[i, j] < 0.0):
                continue

            # --- Emission Calculation ---
            fracland = max(0.0, 1.0 - fraclake[i, j]) * max(0.0, 1.0 - fracsnow[i, j])
            kvh = mb95_vertical_flux_ratio(clay[i, j], kvhmax)
            total_emissions = alpha_grav * fracland * (ssm[i, j] ** gamma) * airdens[i, j] * kvh

            if drag_opt == 1:
                R = rdrag[i, j]
            elif drag_opt == 2:
                R = darmenova_drag_partition(rdrag[i, j], vegfrac[i, j], VEG_THRESHOLD_FENGSHA)
            elif drag_opt == 3:
                R = leung_drag_partition(rdrag[i, j], lai[i, j], vegfrac[i, j], VEG_THRESHOLD_FENGSHA)
            else:
                R = rdrag[i, j]

            rustar = R * ustar[i, j]
            smois = slc[i, j] * moist_correct
            h = gocart_moisture_correction(smois, sand[i, j], clay[i, j], drylimit_factor)
            u_thresh = uthrs[i, j] * h

            u_sum = rustar + u_thresh
            q = max(0.0, rustar - u_thresh) * u_sum * u_sum

            for n in range(nbins):
                emissions[i, j, n] = distribution[n] * total_emissions * q

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
