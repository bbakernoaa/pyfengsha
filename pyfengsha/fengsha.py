"""
NOAA/ARL FENGSHA dust emission model and GOCART2G scheme implementations.
"""

import math
import numpy as np
from numba import jit

# Constants
# Gravity constant used in some internal calculations (cm/s^2 if g0=981).
# However, most functions now take 'grav' as an input.
g0 = 9.81 * 100.0


@jit(nopython=True)
def volumetric_to_gravimetric(vsoil: float, sandfrac: float) -> float:
    """
    Convert volumetric soil moisture to gravimetric.

    Args:
        vsoil: Volumetric Soil Moisture [m3/m3].
        sandfrac: Fractional Sand content [0-1].

    Returns:
        Gravimetric soil moisture [kg/kg].
    """
    soil_dens = 2650.0
    water_dens = 1000.0

    # Saturated volumetric water content (sand-dependent) ! [m3 m-3]
    vsat = 0.489 - 0.00126 * (sandfrac * 100.0)

    # gravimetric soil content
    grav_soil = vsoil * water_dens / (soil_dens * (1.0 - vsat))
    return grav_soil


@jit(nopython=True)
def soilMoistureConvertVol2Grav(volumetricSoilMoisture: float, sandFraction: float) -> float:
    """
    Convert soil moisture fraction from volumetric to gravimetric.

    Args:
        volumetricSoilMoisture: Volumetric soil moisture fraction [1].
        sandFraction: Sand fraction [1].

    Returns:
        Gravimetric soil moisture scaled by 100 (percentage?).
    """
    waterDensity = 1000.0
    particleDensity = 1700.0

    saturatedVolumetricWaterContent = 0.489 - 0.126 * sandFraction

    return 100.0 * volumetricSoilMoisture * waterDensity / (particleDensity * (1.0 - saturatedVolumetricWaterContent))


@jit(nopython=True)
def fecan_dry_limit(clay: float) -> float:
    """
    Calculates the Fecan dry limit.

    Args:
        clay: Fractional clay content.

    Returns:
        Dry limit [kg/kg].
    """
    if clay <= 0.0:
        # 1e-4 used as small epsilon
        drylimit = 14.0 * 1e-4 * 1e-4 + 17.0 * 1e-4
    else:
        drylimit = 14.0 * clay * clay + 17.0 * clay
    return drylimit


@jit(nopython=True)
def fecan_moisture_correction(vol_soil_moisture: float, sand: float, clay: float, b: float = 1.0) -> float:
    """
    Calculates the Fecan soil moisture correction.

    Args:
        vol_soil_moisture: Volumetric soil moisture [m3/m3].
        sand: Fractional sand content.
        clay: Fractional clay content.
        b: Dry limit factor (default 1.0).

    Returns:
        H (correction factor).
    """
    gravsm = volumetric_to_gravimetric(vol_soil_moisture, sand)
    drylimit = 14.0 * clay * clay + 17.0 * clay

    if gravsm > drylimit:
        H = np.sqrt(1.0 + 1.21 * (gravsm - drylimit)**0.68)
    else:
        H = 1.0
    return H


@jit(nopython=True)
def moistureCorrectionFecan(slc: float, sand: float, clay: float, b: float) -> float:
    """
    GOCART version of moisture correction.

    Args:
        slc: Liquid water content of top soil layer, volumetric fraction [1].
        sand: Fractional sand content [1].
        clay: Fractional clay content [1].
        b: Dry limit factor.

    Returns:
        Moisture correction factor.
    """
    gravimetricSoilMoisture = soilMoistureConvertVol2Grav(slc, sand)
    fecanDryLimit = b * clay * (14.0 * clay + 17.0)

    return np.sqrt(1.0 + 1.21 * max(0.0, gravimetricSoilMoisture - fecanDryLimit)**0.68)


@jit(nopython=True)
def shao_1996_soil_moisture(w: float) -> float:
    """
    Shao 1996 soil moisture function.

    Args:
        w: Soil moisture parameter.

    Returns:
        Calculated value.
    """
    return np.exp(22.7 * w)


@jit(nopython=True)
def shao_2004_soil_moisture(w: float) -> float:
    """
    Shao 2004 soil moisture function.

    Args:
        w: Soil moisture parameter.

    Returns:
        Calculated value.
    """
    if w <= 0.03:
        return np.exp(22.7 * w)
    else:
        return np.exp(95.3 * w - 2.029)


@jit(nopython=True)
def modified_threshold(u_ts0: float, H: float, drag: float) -> float:
    """
    Calculate modified threshold velocity.

    Args:
        u_ts0: Threshold friction velocity.
        H: Moisture correction factor.
        drag: Drag partition factor.

    Returns:
        Modified threshold velocity.
    """
    return u_ts0 * H / drag


@jit(nopython=True)
def MB95_kvh(clay: float) -> float:
    """
    Marticorena and Bergametti (1995) vertical to horizontal flux ratio.

    Args:
        clay: Clay fraction.

    Returns:
        Ratio value.
    """
    if clay <= 0.2:
        return 10.0**(13.4 * clay - 6.0)
    else:
        return 2.0E-4


@jit(nopython=True)
def DustFluxV2HRatioMB95(clay_fraction: float, max_flux_ratio: float) -> float:
    """
    Compute dust flux vertical to horizontal ratio (MB95).

    Args:
        clay_fraction: Fraction of clay in soil.
        max_flux_ratio: Maximum flux ratio allowed.

    Returns:
        The computed ratio.
    """
    CLAY_THRESHOLD = 0.2
    if clay_fraction > CLAY_THRESHOLD:
        return max_flux_ratio
    else:
        return 10.0**(13.4 * clay_fraction - 6.0)


@jit(nopython=True)
def fengsha_hflux(ust: float, utst: float) -> float:
    """
    Calculates Horizontal Saltation Flux Q.

    Args:
        ust: Friction velocity.
        utst: Threshold friction velocity.

    Returns:
        Horizontal saltation flux.
    """
    return max(0.0, ust * (ust * ust - utst * utst))


@jit(nopython=True)
def mackinnon_drag(z0: float) -> float:
    """
    MacKinnon drag partition.

    Args:
        z0: Roughness length.

    Returns:
        Drag partition factor.
    """
    z0s = 1.0e-04
    return 1.0 - np.log(z0 / z0s) / np.log(0.7 * (12255.0 / z0s) ** 0.8)


@jit(nopython=True)
def mb95_drag(z0: float) -> float:
    """
    Marticorena and Bergametti (1995) drag partition.

    Args:
        z0: Roughness length.

    Returns:
        Drag partition factor.
    """
    z0s = 1.0e-04
    return 1.0 - np.log(z0 / z0s) / np.log(0.7 * (10.0 / z0s) ** 0.8)


@jit(nopython=True)
def fengsha_albedo(rho_phy: float, smois: float, ssm: float, xland: float, ust: float, clay: float, sand: float, rdrag: float, u_ts0: float) -> float:
    """
    Calculate dust emission using Fengsha albedo based logic.

    Args:
        rho_phy: Air density or similar physical parameter [kg/m3].
        smois: Soil moisture.
        ssm: Surface soil moisture availability?
        xland: Land mask (1 for land).
        ust: Friction velocity.
        clay: Clay fraction.
        sand: Sand fraction.
        rdrag: Drag partition.
        u_ts0: Threshold velocity.

    Returns:
        Dust emission flux.
    """
    g0 = 9.81 * 100.0
    cmb = 1.0

    # Don't do dust over water or where ssm says not possible
    if xland != 1 or ssm <= 0:
        return 0.0

    # soil moisture correction
    # Using original fecan_moisture_correction logic (implicit b=1)
    H = fecan_moisture_correction(smois, sand, clay)

    # vertical to horizontal mass flux
    kvh = MB95_kvh(clay)

    # modified threshold velocity
    u_ts = modified_threshold(u_ts0, H, rdrag)

    # horizontal mass flux
    ustar_albedo = ust * rdrag
    Q = fengsha_hflux(ustar_albedo, u_ts)

    emis_dust = cmb * ssm * rho_phy / g0 * kvh * Q
    return emis_dust


@jit(nopython=True)
def fengsha(rho_phy: float, smois: float, ssm: float, xland: float, ust: float, clay: float, sand: float, rdrag: float, u_ts0: float) -> float:
    """
    Core Fengsha dust emission calculation.

    Args:
        rho_phy: Physical density parameter.
        smois: Soil moisture.
        ssm: Surface soil moisture.
        xland: Land mask.
        ust: Friction velocity.
        clay: Clay fraction.
        sand: Sand fraction.
        rdrag: Drag partition.
        u_ts0: Threshold velocity.

    Returns:
        Dust emission.
    """
    g0 = 9.81 * 100.0
    cmb = 1.0

    if xland != 1 or ssm <= 0:
        return 0.0

    H = fecan_moisture_correction(smois, sand, clay)
    kvh = MB95_kvh(clay)
    u_ts = modified_threshold(u_ts0, H, rdrag)
    Q = fengsha_hflux(ust, u_ts)

    emis_dust = cmb * ssm * rho_phy / g0 * kvh * Q
    return emis_dust


@jit(nopython=True)
def DustAerosolDistributionKok(radius: np.ndarray, rLow: np.ndarray, rUp: np.ndarray) -> np.ndarray:
    """
    Compute Kok's dust size aerosol distribution.

    Args:
        radius: Array of bin radii.
        rLow: Lower bounds of bins.
        rUp: Upper bounds of bins.

    Returns:
        Array of distribution fractions summing to 1.
    """
    medianMassDiameter = 3.4
    geometricStdDev = 3.0
    crackPropagationLength = 12.0
    factor = 1.0 / (np.sqrt(2.0) * np.log(geometricStdDev))

    numBins = len(radius)
    distribution = np.zeros(numBins)
    totalVolume = 0.0

    for n in range(numBins):
        diameter = 2.0 * radius[n]
        dlam = diameter / crackPropagationLength
        distribution[n] = diameter * (1.0 + math.erf(factor * np.log(diameter / medianMassDiameter))) * \
                          np.exp(-dlam**3) * np.log(rUp[n] / rLow[n])
        totalVolume += distribution[n]

    for n in range(numBins):
        distribution[n] = distribution[n] / totalVolume

    return distribution


@jit(nopython=True)
def DarmenovaDragPartition(Lc: float, vegfrac: float, thresh: float) -> float:
    """
    Darmenova drag partition scheme.

    Args:
        Lc: Roughness density?
        vegfrac: Vegetation fraction.
        thresh: Threshold value.

    Returns:
        Effective drag partition.
    """
    DRAG_MIN = 1.0e-3
    sigb = 1.0
    mb = 0.5
    Betab = 90.0
    sigv = 1.45
    mv = 0.16
    Betav = 202.0

    if vegfrac < 0.0 or vegfrac >= thresh:
        feff_veg = DRAG_MIN
    else:
        Lc_veg = -0.35 * np.log(1.0 - vegfrac)
        # calc_drag_partition inline
        R1 = 1.0 / np.sqrt(1.0 - sigv * mv * Lc_veg)
        R2 = 1.0 / np.sqrt(1.0 + mv * Betav * Lc_veg)
        feff_veg = R1 * R2

    Lc_bare = Lc / (1.0 - vegfrac)
    tmpVal = 1.0 - sigb * mb * Lc_bare

    skip_bare = False
    if vegfrac < 0.0 or vegfrac >= thresh:
        skip_bare = True
    elif (Lc > 0.2) or (tmpVal <= 0.0):
        skip_bare = True

    if not skip_bare:
        R1 = 1.0 / np.sqrt(1.0 - sigb * mb * Lc_bare)
        R2 = 1.0 / np.sqrt(1.0 + mb * Betab * Lc_bare)
        feff_bare = R1 * R2
    else:
        feff_bare = DRAG_MIN

    feff = feff_veg * feff_bare

    if feff > 1.0 or feff < 1.0e-5:
        return DRAG_MIN
    else:
        return feff


@jit(nopython=True)
def LeungDragPartition(Lc: float, lai: float, gvf: float, thresh: float) -> float:
    """
    Leung drag partition scheme.

    Args:
        Lc: Roughness density.
        lai: Leaf Area Index.
        gvf: Green Vegetation Fraction.
        thresh: Threshold value.

    Returns:
        Effective drag partition.
    """
    LAI_THR = 0.33
    C = 4.8
    F0 = 0.32
    SIGB = 1.0
    MB = 0.5
    BETAB = 90.0
    MIN_FEFF = 1.0E-5
    MAX_FEFF = 1.0
    SMALL = 1.0E-10

    frac_bare = max(min(1.0 - lai / thresh, 1.0), SMALL)

    if (lai <= 0.0) or (lai >= thresh):
        feff_veg = 0.0
    else:
        K = 2.0 * (1.0 / max(1.0 - lai, SMALL) - 1.0)
        feff_veg = (K + F0 * C) / (K + C)

    if (Lc <= 0.2) and (Lc > 0.0) and (lai < thresh):
        Lc_bare = Lc / max(frac_bare, SMALL)
        tmpVal = 1.0 - SIGB * MB * Lc_bare

        if tmpVal > SMALL:
            Rbare1 = 1.0 / np.sqrt(max(1.0 - SIGB * MB * Lc_bare, SMALL))
            Rbare2 = 1.0 / np.sqrt(1.0 + BETAB * MB * Lc_bare)
            feff_bare = Rbare1 * Rbare2
        else:
            feff_bare = 0.0
    else:
        feff_bare = 0.0

    feff = (gvf * feff_veg**3 + frac_bare * feff_bare**3) ** (1.0/3.0)

    if feff > MAX_FEFF or feff < MIN_FEFF:
        return MIN_FEFF
    else:
        return feff


@jit(nopython=True)
def DustEmissionFENGSHA(fraclake: np.ndarray, fracsnow: np.ndarray, oro: np.ndarray, slc: np.ndarray, clay: np.ndarray, sand: np.ndarray, silt: np.ndarray,
                        ssm: np.ndarray, rdrag: np.ndarray, airdens: np.ndarray, ustar: np.ndarray, vegfrac: np.ndarray, lai: np.ndarray, uthrs: np.ndarray,
                        alpha: float, gamma: float, kvhmax: float, grav: float, rhop: np.ndarray, distribution: np.ndarray,
                        drylimit_factor: float, moist_correct: float, drag_opt: int) -> np.ndarray:
    """
    Compute dust emissions using NOAA/ARL FENGSHA model.

    Args:
        fraclake: Fraction of lake coverage.
        fracsnow: Fraction of snow coverage.
        oro: Land/water mask (1 for land).
        slc: Soil liquid content.
        clay: Clay fraction.
        sand: Sand fraction.
        silt: Silt fraction.
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
        rhop: Particle density per bin.
        distribution: Size distribution per bin.
        drylimit_factor: Dry limit factor for moisture correction.
        moist_correct: Moisture correction factor.
        drag_opt: Drag option (1, 2, or 3).

    Returns:
        Emissions array of shape (ni, nj, nbins).
    """
    SSM_THRESH = 1.0E-02
    VEG_THRESH = 0.4
    SMALL = 1.0E-10
    MAX_RDRAG = 0.3
    LAND = 1.0

    ni = fraclake.shape[0]
    nj = fraclake.shape[1]
    nbins = len(distribution)

    emissions = np.zeros((ni, nj, nbins))

    alpha_grav = alpha / max(grav, SMALL)

    for j in range(nj):
        for i in range(ni):
            skip = False
            if oro[i, j] != LAND:
                skip = True

            if not skip:
                if drag_opt == 2:
                    if (vegfrac[i, j] < 0.0) or (vegfrac[i, j] >= VEG_THRESH) or (rdrag[i, j] > MAX_RDRAG):
                        skip = True
                elif drag_opt == 3:
                    if (vegfrac[i, j] < 0.0) or (lai[i, j] >= VEG_THRESH):
                        skip = True
                else:
                    if rdrag[i, j] < 0.0:
                        skip = True

            if not skip:
                if (ssm[i, j] < SSM_THRESH) or (clay[i, j] < 0.0) or (sand[i, j] < 0.0):
                    skip = True

            if not skip:
                fracland = max(0.0, min(1.0, 1.0 - fraclake[i, j])) * \
                           max(0.0, min(1.0, 1.0 - fracsnow[i, j]))

                kvh = DustFluxV2HRatioMB95(clay[i, j], kvhmax)

                total_emissions = alpha_grav * fracland * (ssm[i, j] ** gamma) * \
                                  airdens[i, j] * kvh

                if drag_opt == 1:
                    R = rdrag[i, j]
                elif drag_opt == 2:
                    R = DarmenovaDragPartition(rdrag[i, j], vegfrac[i, j], VEG_THRESH)
                elif drag_opt == 3:
                    R = LeungDragPartition(rdrag[i, j], lai[i, j], vegfrac[i, j], VEG_THRESH)
                else:
                    R = rdrag[i, j] # Default

                rustar = R * ustar[i, j]

                smois = slc[i, j] * moist_correct
                h = moistureCorrectionFecan(smois, sand[i, j], clay[i, j], drylimit_factor)

                u_thresh = uthrs[i, j] * h
                u_sum = rustar + u_thresh

                q = max(0.0, rustar - u_thresh) * u_sum * u_sum

                for n in range(nbins):
                    emissions[i, j, n] = distribution[n] * total_emissions * q

    return emissions


@jit(nopython=True)
def DustEmissionGOCART2G(radius: np.ndarray, fraclake: np.ndarray, gwettop: np.ndarray, oro: np.ndarray, u10m: np.ndarray, v10m: np.ndarray, Ch_DU: float, du_src: np.ndarray, grav: float) -> np.ndarray:
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
    soil_density = 2650.0
    LAND = 1.0

    nbins = len(radius)
    ni = u10m.shape[0]
    nj = u10m.shape[1]

    emissions = np.zeros((ni, nj, nbins))

    for n in range(nbins):
        diameter = 2.0 * radius[n]

        u_thresh0 = 0.13 * np.sqrt(soil_density * grav * diameter / air_dens) * \
                    np.sqrt(1.0 + 6.0e-7 / (soil_density * grav * diameter**2.5)) / \
                    np.sqrt(1.928 * (1331.0 * (100.0 * diameter)**1.56 + 0.38)**0.092 - 1.0)

        for j in range(nj):
            for i in range(ni):
                if oro[i, j] != LAND:
                    continue

                w10m = np.sqrt(u10m[i, j]**2 + v10m[i, j]**2)

                if gwettop[i, j] < 0.5:
                    u_thresh = max(0.0, u_thresh0 * (1.2 + 0.2 * np.log10(max(1.e-3, gwettop[i, j]))))

                    if w10m > u_thresh:
                        emissions[i, j, n] = (1.0 - fraclake[i, j]) * w10m**2 * (w10m - u_thresh)

        # Apply scaling and source function
        for j in range(nj):
            for i in range(ni):
                emissions[i, j, n] = Ch_DU * du_src[i, j] * emissions[i, j, n]

    return emissions
