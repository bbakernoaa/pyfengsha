import numpy as np
from numba import jit
import math

# Constants
g0 = 9.81 * 100 # gravity in cm/s^2? Original code says "gravity", updated code uses "grav" as input.
# In original fengsha.F90: real, parameter :: g0 = 9.81 * 100 ! gravity
# This seems to be cm/s^2 if 9.81 is m/s^2. But usually gravity is 9.81 m/s^2.
# Let's check usage. emis_dust = cmb * ssm * rho_phy / g0 * kvh * Q
# rho_phy [kg/m3]. Q [g/m**2-s] * ??
# In new code: grav is input [m/s^2].
# I will implement functions exactly as they logic dictates.

@jit(nopython=True)
def volumetric_to_gravimetric(vsoil, sandfrac):
    """
    Convert volumetric soil moisture to gravimetric.
    vsoil: Volumetric Soil Moisture [m3/m3]
    sandfrac: fractional Sand content [0-1]
    Returns: grav_soil [kg/kg]
    """
    soil_dens = 2650.0
    water_dens = 1000.0

    # Saturated volumetric water content (sand-dependent) ! [m3 m-3]
    vsat = 0.489 - 0.00126 * (sandfrac * 100.0)

    # gravimetric soil content
    grav_soil = vsoil * water_dens / (soil_dens * (1.0 - vsat))
    return grav_soil

@jit(nopython=True)
def soilMoistureConvertVol2Grav(volumetricSoilMoisture, sandFraction):
    """
    Convert soil moisture fraction from volumetric to gravimetric.
    volumetricSoilMoisture: [1]
    sandFraction: [1]
    Returns: gravimetric soil moisture [kg/kg] * 100? No, let's check GOCART implementation.

    GOCART:
    soilMoistureConvertVol2Grav = 100.0 * volumetricSoilMoisture * waterDensity / (particleDensity * (1.0 - saturatedVolumetricWaterContent))
    It returns percentage?
    original volumetric_to_gravimetric: vsoil * water_dens / (soil_dens * (1. - vsat))
    The original one doesn't have 100.0 factor.
    The GOCART one has 100.0 factor.
    """
    waterDensity = 1000.0
    particleDensity = 1700.0 # Note: different density in GOCART (1700) vs original (2650)

    # Compute saturated volumetric water content (sand-dependent) [m3 m-3]
    # GOCART: 0.489 - 0.126 * sandFraction. (Assuming sandFraction is 0-1)
    # Original: 0.489 - 0.00126 * (sandfrac * 100). Same.
    saturatedVolumetricWaterContent = 0.489 - 0.126 * sandFraction

    # Convert volumetric soil moisture to gravimetric soil moisture
    # GOCART code returns: 100.0 * ...
    return 100.0 * volumetricSoilMoisture * waterDensity / (particleDensity * (1.0 - saturatedVolumetricWaterContent))

@jit(nopython=True)
def fecan_dry_limit(clay):
    """
    Calculates the fecan dry limit.
    clay: fractional clay content
    Returns: drylimit [kg/kg]
    """
    if clay <= 0.0:
        # 1e-4 used as small epsilon
        drylimit = 14.0 * 1e-4 * 1e-4 + 17.0 * 1e-4
    else:
        drylimit = 14.0 * clay * clay + 17.0 * clay
    return drylimit

@jit(nopython=True)
def fecan_moisture_correction(vol_soil_moisture, sand, clay, b=1.0):
    """
    Calculates the fecan soil moisture correction.
    vol_soil_moisture: volumetric soil moisture [m3/m3]
    sand: fractional sand content
    clay: fractional clay content
    b: drylimit factor (default 1.0 in original code, but passed in GOCART version)
    Returns: H (correction factor)
    """
    # Note: original code uses b=1.0 implicitly (doesn't have it as arg), but defined parameter?
    # Actually original code calculates drylimit inside.
    # GOCART version passes 'b'.

    gravsm = volumetric_to_gravimetric(vol_soil_moisture, sand)

    # fecan dry limit
    # original: drylimit=14.0*clay*clay+17.0*clay
    drylimit = 14.0 * clay * clay + 17.0 * clay

    # fecan soil moisture correction
    if gravsm > drylimit:
        H = np.sqrt(1.0 + 1.21 * (gravsm - drylimit)**0.68)
    else:
        H = 1.0
    return H

@jit(nopython=True)
def moistureCorrectionFecan(slc, sand, clay, b):
    """
    GOCART version of moisture correction.
    slc: liquid water content of top soil layer, volumetric fraction [1]
    sand: fractional sand content [1]
    clay: fractional clay content [1]
    b: drylimit factor
    """
    # Convert soil moisture from volumetric to gravimetric
    # Note: GOCART uses soilMoistureConvertVol2Grav which returns value scaled by 100?
    # Let's check GOCART usage.
    # gravimetricSoilMoisture = soilMoistureConvertVol2Grav(slc, sand)
    # fecanDryLimit = b * clay * (14.0 * clay + 17.0)
    # moistureCorrectionFecan = sqrt(1.0 + 1.21 * max(0.0, gravimetricSoilMoisture - fecanDryLimit)**0.68)

    gravimetricSoilMoisture = soilMoistureConvertVol2Grav(slc, sand)
    fecanDryLimit = b * clay * (14.0 * clay + 17.0)

    return np.sqrt(1.0 + 1.21 * max(0.0, gravimetricSoilMoisture - fecanDryLimit)**0.68)

@jit(nopython=True)
def shao_1996_soil_moisture(w):
    return np.exp(22.7 * w)

@jit(nopython=True)
def shao_2004_soil_moisture(w):
    if w <= 0.03:
        return np.exp(22.7 * w)
    else:
        return np.exp(95.3 * w - 2.029)

@jit(nopython=True)
def modified_threshold(u_ts0, H, drag):
    return u_ts0 * H / drag

@jit(nopython=True)
def MB95_kvh(clay):
    if clay <= 0.2:
        return 10.0**(13.4 * clay - 6.0)
    else:
        return 2.0E-4

@jit(nopython=True)
def DustFluxV2HRatioMB95(clay_fraction, max_flux_ratio):
    CLAY_THRESHOLD = 0.2
    if clay_fraction > CLAY_THRESHOLD:
        return max_flux_ratio
    else:
        return 10.0**(13.4 * clay_fraction - 6.0)

@jit(nopython=True)
def fengsha_hflux(ust, utst):
    """
    Calculates Horizontal Saltation Flux Q.
    """
    return max(0.0, ust * (ust * ust - utst * utst))

@jit(nopython=True)
def mackinnon_drag(z0):
    z0s = 1.0e-04
    return 1.0 - np.log(z0 / z0s) / np.log(0.7 * (12255.0 / z0s) ** 0.8)

@jit(nopython=True)
def mb95_drag(z0):
    z0s = 1.0e-04
    return 1.0 - np.log(z0 / z0s) / np.log(0.7 * (10.0 / z0s) ** 0.8)

@jit(nopython=True)
def fengsha_albedo(rho_phy, smois, ssm, xland, ust, clay, sand, rdrag, u_ts0):
    """
    fengsha_albedo implementation.
    """
    g0 = 9.81 * 100.0 # cm/s^2 ?
    cmb = 1.0

    # Don't do dust over water or where ssm says not possible
    if xland != 1 or ssm <= 0:
        return 0.0

    rhoa = rho_phy * 1.0E-3

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
def fengsha(rho_phy, smois, ssm, xland, ust, clay, sand, rdrag, u_ts0):
    """
    fengsha implementation.
    """
    g0 = 9.81 * 100.0
    cmb = 1.0

    if xland != 1 or ssm <= 0:
        return 0.0

    rhoa = rho_phy * 1.0E-3
    H = fecan_moisture_correction(smois, sand, clay)
    kvh = MB95_kvh(clay)
    u_ts = modified_threshold(u_ts0, H, rdrag)
    Q = fengsha_hflux(ust, u_ts)

    emis_dust = cmb * ssm * rho_phy / g0 * kvh * Q
    return emis_dust

@jit(nopython=True)
def DustAerosolDistributionKok(radius, rLow, rUp):
    """
    Compute Kok's dust size aerosol distribution
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
        # Using erf from math (std python) inside numba, or numpy if available
        # Numba supports math.erf
        distribution[n] = diameter * (1.0 + math.erf(factor * np.log(diameter / medianMassDiameter))) * \
                          np.exp(-dlam**3) * np.log(rUp[n] / rLow[n])
        totalVolume += distribution[n]

    for n in range(numBins):
        distribution[n] = distribution[n] / totalVolume

    return distribution

@jit(nopython=True)
def DarmenovaDragPartition(Lc, vegfrac, thresh):
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
def LeungDragPartition(Lc, lai, gvf, thresh):
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
def DustEmissionFENGSHA(fraclake, fracsnow, oro, slc, clay, sand, silt,
                        ssm, rdrag, airdens, ustar, vegfrac, lai, uthrs,
                        alpha, gamma, kvhmax, grav, rhop, distribution,
                        drylimit_factor, moist_correct, drag_opt):
    """
    Compute dust emissions using NOAA/ARL FENGSHA model.
    Returns emissions array of shape (ni, nj, nbins).
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
def DustEmissionGOCART2G(radius, fraclake, gwettop, oro, u10m, v10m, Ch_DU, du_src, grav):
    """
    Computes dust emissions using GOCART2G scheme.
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

@jit(nopython=True)
def VerticalDustFluxK14_single(u, u_t, rho_air, f_erod, k_gamma):
    """
    Helper for VerticalDustFluxK14
    """
    rho_a0 = 1.225
    u_st0 = 0.16
    C_d0 = 4.4e-5
    C_e = 2.0
    C_a = 2.7

    emission = 0.0

    if (f_erod > 0.0) and (u > u_t):
        u_st = u_t * np.sqrt(rho_air / rho_a0)
        u_st = max(u_st, u_st0)

        f_ust = (u_st - u_st0) / u_st0
        C_d = C_d0 * np.exp(-C_e * f_ust)

        emission = C_d * f_erod * k_gamma * rho_air * \
                   ((u*u - u_t*u_t) / u_st) * \
                   (u / u_t)**(C_a * f_ust)

    return emission

@jit(nopython=True)
def VerticalDustFluxK14(u, u_t, rho_air, f_erod, k_gamma):
    ni = u.shape[0]
    nj = u.shape[1]
    emissions = np.zeros((ni, nj))

    for j in range(nj):
        for i in range(ni):
            emissions[i, j] = VerticalDustFluxK14_single(u[i, j], u_t[i, j], rho_air[i, j],
                                                         f_erod[i, j], k_gamma[i, j])
    return emissions

@jit(nopython=True)
def DustEmissionK14(t_soil, w_top, rho_air, z0, z, u_z, v_z, ustar,
                    f_land, f_snow, f_src, f_sand, f_silt, f_clay,
                    texture, vegetation, gvf,
                    f_w, f_c, uts_gamma, UNDEF, GRAV, VON_KARMAN,
                    opt_clay, Ch_DU):
    """
    DustEmissionK14 implementation.
    Returns: emissions, u, u_t, u_ts, R, H_w, f_erod
    """
    ni = t_soil.shape[0]
    nj = t_soil.shape[1]

    # Allocating outputs
    # emissions size 3rd dim is not specified in inputs, but GOCART code uses size(emissions, 3).
    # Since we return it, we need to know. The original code loops `n=2, size...`.
    # It assumes bin 1 is calculated and broadcasted.
    # We will return the single bin emission map, and let Python handle broadcasting if needed.
    # Or we can return (ni, nj, 1) or whatever.
    # The prompt listed `emissions(:,:,:)` as output.
    # Let's assume 1 bin for now, or maybe passed as argument?
    # In GOCART code: `emissions(:,:,1)` is computed.

    z0_valid = 0.08e-2
    z0_max = 6.25 * z0_valid
    rho_water = 1000.0
    rho_soil = 2500.0
    a_n = 0.0123
    Dp_size = 75e-6
    rho_p = 2.65e3

    # Dc_soil is 12 elements.
    Dc_soil = np.array([710e-6, 710e-6, 125e-6, 125e-6, 125e-6, 160e-6,
                        710e-6, 125e-6, 125e-6, 160e-6, 125e-6, 2e-6])

    u_ts = np.full((ni, nj), UNDEF)
    for j in range(nj):
        for i in range(ni):
            if f_land[i, j] > 0.0 and z0[i, j] < z0_max and z0[i, j] > 0.0:
                u_ts[i, j] = np.sqrt(a_n * ((rho_p/rho_air[i, j]) * GRAV * Dp_size + uts_gamma / (rho_air[i, j] * Dp_size)))

    # w_g
    w_g = np.full((ni, nj), UNDEF)
    for j in range(nj):
        for i in range(ni):
            if f_land[i, j] > 0.0:
                w_g[i, j] = 100.0 * f_w * rho_water / rho_soil / (1.0 - (0.489 - 0.126 * f_sand[i, j])) * w_top[i, j]

    # clay, silt, w_gt
    clay_corr = np.full((ni, nj), UNDEF)
    silt_corr = np.full((ni, nj), UNDEF)
    w_gt = np.full((ni, nj), UNDEF)

    for j in range(nj):
        for i in range(ni):
            if f_land[i, j] > 0.0 and f_clay[i, j] <= 1.0 and f_clay[i, j] >= 0.0:
                clay_corr[i, j] = f_c * f_clay[i, j]
                silt_corr[i, j] = f_silt[i, j] + (1.0 - f_c) * f_clay[i, j]
                w_gt[i, j] = 14.0 * clay_corr[i, j] * clay_corr[i, j] + 17.0 * clay_corr[i, j]

    H_w = np.ones((ni, nj))
    for j in range(nj):
        for i in range(ni):
            if f_land[i, j] > 0.0 and w_g[i, j] > w_gt[i, j]:
                H_w[i, j] = np.sqrt(1.0 + 1.21 * (w_g[i, j] - w_gt[i, j])**0.68)

    k_gamma = np.zeros((ni, nj))
    if opt_clay == 1:
        for j in range(nj):
            for i in range(ni):
                if f_land[i, j] > 0.0:
                    c = clay_corr[i, j]
                    if c < 0.2 and c >= 0.05:
                        k_gamma[i, j] = c
                    elif c >= 0.2 and c <= 1.0:
                        k_gamma[i, j] = 0.2
                    else:
                        k_gamma[i, j] = 0.05
    elif opt_clay == 2:
        for j in range(nj):
            for i in range(ni):
                if f_land[i, j] > 0.0:
                    c = clay_corr[i, j]
                    s = silt_corr[i, j]
                    if c < 0.2 and c >= 0.0:
                        k_gamma[i, j] = 1.0 / (1.4 - c - s)
                    elif c >= 0.2 and c <= 1.0:
                        k_gamma[i, j] = 1.0 / (1.0 + c - s)
                    else:
                        k_gamma[i, j] = 1.0 / 1.4
    else:
        for j in range(nj):
            for i in range(ni):
                if f_land[i, j] > 0.0 and clay_corr[i, j] <= 1.0 and clay_corr[i, j] >= 0.0:
                    k_gamma[i, j] = clay_corr[i, j]

    z0s = np.full((ni, nj), 125e-6)
    for j in range(nj):
        for i in range(ni):
            tex = int(round(texture[i, j]))
            if tex > 0 and tex < 13:
                z0s[i, j] = Dc_soil[tex-1]
            z0s[i, j] = z0s[i, j] / 30.0

    R = np.ones((ni, nj))
    for j in range(nj):
        for i in range(ni):
            if f_land[i, j] > 0.0 and z0[i, j] < z0_max and z0[i, j] > z0s[i, j]:
                R[i, j] = 1.0 - np.log(z0[i, j] / z0s[i, j]) / np.log(0.7 * (122.55 / z0s[i, j])**0.8)

    u = np.full((ni, nj), UNDEF)
    for j in range(nj):
        for i in range(ni):
            if f_land[i, j] > 0.0 and z0[i, j] < z0_max and z0[i, j] > 0.0:
                u[i, j] = ustar[i, j] * R[i, j]

    u_t = np.full((ni, nj), UNDEF)
    for j in range(nj):
        for i in range(ni):
            if f_land[i, j] > 0.0 and z0[i, j] < z0_max and z0[i, j] > 0.0:
                u_t[i, j] = u_ts[i, j] * H_w[i, j]

    f_erod = np.full((ni, nj), UNDEF)
    for j in range(nj):
        for i in range(ni):
            if f_land[i, j] > 0.0:
                f_erod[i, j] = 1.0
                if z0[i, j] > 3.0e-5 and z0[i, j] < z0_max:
                    f_erod[i, j] = 0.7304 - 0.0804 * np.log10(100.0 * z0[i, j])

                if abs(texture[i, j] - 15) < 0.5:
                    f_erod[i, j] = 0.0

    f_veg = np.zeros((ni, nj))
    for j in range(nj):
        for i in range(ni):
            if f_land[i, j] > 0.0:
                # Assuming vegetation categories check
                if abs(vegetation[i, j] - 7) < 0.1:
                    f_veg[i, j] = 1.0
                if abs(vegetation[i, j] - 16) < 0.1:
                    f_veg[i, j] = 1.0

            if f_land[i, j] > 0.0 and gvf[i, j] >= 0.0 and gvf[i, j] < 0.8:
                f_veg[i, j] = f_veg[i, j] * (1.0 - gvf[i, j])

            f_erod[i, j] = f_erod[i, j] * f_veg[i, j] * f_land[i, j] * (1.0 - f_snow[i, j])

            if f_src[i, j] >= 0.0:
                f_erod[i, j] = f_src[i, j] * f_erod[i, j]

    emission_slab = VerticalDustFluxK14(u, u_t, rho_air, f_erod, k_gamma)
    emission_slab = emission_slab * Ch_DU

    return emission_slab, u, u_t, u_ts, R, H_w, f_erod
