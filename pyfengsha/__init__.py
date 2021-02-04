import fengsha as fdust


def fengsha_albedo(rhoa, volumetric_soil_moisture, ssm, land, ustar, clayfrac, sandfrac, drag_partition, dry_threshold):
    """This calculates the total flux using the modified fengsha parameterization for surface friction based on
        Chappel and Webb 2017

    Parameters
    ----------
    rhoa : float
        surface air density [kg/m3]
    volumetric_soil_moisture : float
        volumetric soil moisture [m3/m3]
    ssm : float
        Sediment Supply Map [-]
    land : float
        Land or water flag [1 for land 0 for water]
    ustar : float
        Boundary layer friction velocity
    clayfrac : float
        Fractional clay content [-] : range 0->1
    sandfrac : float
        Fractional sand content [-] : range 0->1
    drag_partition : float
        drag partition [-]
    dry_threshold : float
        Dry Threshold friction velocity [m/s]

    Returns
    -------
    float
        Total mass emitted [g/s]

    """
    emission = fdust.fengsha_albedo(rhoa, volumetric_soil_moisture, ssm, land, ustar, clayfrac, sandfrac, drag_partition, dry_threshold)
    return emission


def fengsha(rhoa, volumetric_soil_moisture, ssm, land, ustar, clayfrac, sandfrac, drag_partition, dry_threshold):
    """This calculates the total flux using the fengsha parameterization

    Parameters
    ----------
    rhoa : float
        surface air density [kg/m3]
    volumetric_soil_moisture : float
        volumetric soil moisture [m3/m3]
    ssm : float
        Sediment Supply Map [-]
    land : float
        Land or water flag [1 for land 0 for water]
    ustar : float
        Boundary layer friction velocity
    clayfrac : float
        Fractional clay content [-] : range 0->1
    sandfrac : float
        Fractional sand content [-] : range 0->1
    drag_partition : float
        drag partition [-]
    dry_threshold : float
        Dry Threshold friction velocity [m/s]

    Returns
    -------
    float
        Total mass emitted [g/s]

    """
    emission = fdust.fengsha(rhoa, volumetric_soil_moisture, ssm, land, ustar, clayfrac, sandfrac, drag_partition, dry_threshold)
    return emission


def mackinnon_drag_partition(z0):
    """Mackinnon Drag partition

      ! ------------------------------------------------------------------------
      !
      !   R = 1.0 - log(z0 / z0s) / log( 0.7 * (12255./z0s) ** 0.8)
      !
      !    Drag partition correction. See MacKinnon et al. (2004),
      !       doi:10.1016/j.geomorph.2004.03.009
      !
      !--------------------------------------------------------------------------

    Parameters
    ----------
    z0 : float
        surface roughness

    Returns
    -------
    float
        mackinnon drag partition

    """
    return fdust.mackinnon_drag(z0)


def mb95_drag_partition(z0):
    """MB95 Drag partition

      ! ------------------------------------------------------------------------
      ! Drag partition correction. See Marticorena et al. (1997),
      !     doi:10.1029/96JD02964
      ! R = 1.0 - log(z0 / z0s) / log( 0.7 * (10./z0s) ** 0.8)
      ! ------------------------------------------------------------------------

    Parameters
    ----------
    z0 : float
        surface roughness

    Returns
    -------
    float
        mackinnon drag partition

    """
    return fdust.mb95_drag(z0)


def draxler_hflux(ustar, threshold_velocity):
    """This calculates the horizontal dust emission flux

    See Draxler & Gillette (2001) Atmos. Environ.

    Q = u*^3 ( 1 - [ ut* / u* ]^2 )

    where

    ut* = u_t0* H / R

    Parameters
    ----------
    ustar : float
        Boundary layer friction velocity [m/s]
    threshold_velocity : float
        modified threshold velocity [m/s]

    Returns
    -------
    float
        Total mass emitted [g/s]

    """
    return fdust.fengsha_hflux(ustar, threshold_velocity)


def mb95_kvh(clay):
    """MB95 Vertical to horizontal mass flux ratio

      !---------------------------------------------------------------------
      ! Function: Calculates the vertical to horizontal mass flux ratio.
      !
      ! formula of MB95
      ! kvh = 10.0**(13.4*clay-6.0)
      !
      !--------------------------------------------------------------------

    Parameters
    ----------
    clay : float
        fractional clay content

    Returns
    -------
    float
        MB95 Vertical to horizontal mass flux ratio

    """
    return fdust.mb95_kvh(clay)


def fecan_moisture_correction(volumetric_soil_moisture, sandfrac, clayfrac):
    """Fecan soil moisture

    !---------------------------------------------------------------------
    ! drylimit = 14.0*clay*clay+17.0*clay
    ! H = sqrt(1.0 + 1.21*(gravsm-drylimit)**0.68)
    !---------------------------------------------------------------------

    Parameters
    ----------
    volumetric_soil_moisture : float
        volumetric_soil_moisture [m3/m3]
    sandfrac : type
        fractional sand content [-]
    clayfrac : type
        fractional clay content [-]

    Returns
    -------
    float
        H : Soil moisture correction factor

    """
    return fdust.fecan_moisture_correction(volumetric_soil_moisture, sandfrac, clayfrac)


def fecan_dry_limit(clayfrac):
    """Short summary.

    Parameters
    ----------
    clayfrac : type
        fractional clay content [-]

    Returns
    -------
    type
        Description of returned object.

    """
    return fdust.fecan_dry_limit(clayfrac)


def volumetric_to_gravimetric(volumetric_soil_moisture, sandfrac):
    """Short summary.

    Parameters
    ----------
    volumetric_soil_moisture : float
        volumetric_soil_moisture [m3/m3]
    sandfrac : type
        fractional sand content [-]

    Returns
    -------
    float
        H : Soil moisture correction factor

    """
    return fdust.volumetric_soil_moisture(volumetric_soil_moisture, sandfrac)


def shao_1996_soil_moisture(volumetric_soil_moisture):
    """Calculates the soil moisture correction factor from Shao 1996

    Parameters
    ----------
    volumetric_soil_moisture : float
        volumetric_soil_moisture [m3/m3]

    Returns
    -------
    float
        H : Soil moisture correction factor

    """
    return fdust.shao_1996_soil_moisture(volumetric_soil_moisture)


def shao_2004_soil_moisture(volumetric_soil_moisture):
    """Calculates the soil moisture correction factor from Shao 2004

    Parameters
    ----------
    volumetric_soil_moisture : float
        volumetric_soil_moisture [m3/m3]

    Returns
    -------
    float
        Description of returned object.

    """
    return fdust.shao_2004_soil_moisture(volumetric_soil_moisture)


def modified_threshold_velocity(dry_threshold, moisture_correction, drag_partition):
    """Calculate the modified threshold friction velocity

    !----------------------------------------------------------
    ! ut* = u_t0* H / R
    !----------------------------------------------------------

    Parameters
    ----------
    dry_threshold : float
        dry threshold friction velocity [m/s]
    moisture_correction : float
        moisture correction factor [-]
    drag_partition : float
        drag partition [-1]

    Returns
    -------
    float
        modified threshold friction velocity

    """
    return fdust.modified_threshold(dry_threshold, moisture_correction, drag_partition)


def xarray_fengsha_albedo(rhoa, volumetric_soil_moisture, ssm, land, ustar, clayfrac, sandfrac, drag_partition, dry_threshold):
    """This function applies the fengsha_albedo function onto an 2d xarray object (ie gridded files)

    Parameters
    ----------
    rhoa : xarray.DataArray
        surface air density [kg/m3]
    volumetric_soil_moisture : xarray.DataArray
        volumetric soil moisture [m3/m3]
    ssm : xarray.DataArray
        Sediment Supply Map [-]
    land : xarray.DataArray
        Land or water flag [1 for land 0 for water]
    ustar : xarray.DataArray
        Boundary layer friction velocity
    clayfrac : xarray.DataArray
        Fractional clay content [-] : range 0->1
    sandfrac : xarray.DataArray
        Fractional sand content [-] : range 0->1
    drag_partition : xarray.DataArray
        drag partition [-]
    dry_threshold : xarray.DataArray
        Dry Threshold friction velocity [m/s]

    Returns
    -------
    xarray.DataArray
        Total mass emitted [g/s]
    """
    func = lambda a, b, c, d, e, f, g, h, i: pyfengsha.fengsha_albedo(a, b, c, d, e, f, g, h, i)
    return xr.apply_ufunc(func, rhoa, volumetric_soil_moisture, ssm, land, ustar, clayfrac, sandfrac, drag_partition, dry_threshold, vectorize=True)


def xarray_fengsha(rhoa, volumetric_soil_moisture, ssm, land, ustar, clayfrac, sandfrac, drag_partition, dry_threshold):
    """This function applies the fenghsa function onto an 2d xarray object (ie gridded files)

    Parameters
    ----------
    rhoa : xarray.DataArray
        surface air density [kg/m3]
    volumetric_soil_moisture : xarray.DataArray
        volumetric soil moisture [m3/m3]
    ssm : xarray.DataArray
        Sediment Supply Map [-]
    land : xarray.DataArray
        Land or water flag [1 for land 0 for water]
    ustar : xarray.DataArray
        Boundary layer friction velocity
    clayfrac : xarray.DataArray
        Fractional clay content [-] : range 0->1
    sandfrac : xarray.DataArray
        Fractional sand content [-] : range 0->1
    drag_partition : xarray.DataArray
        drag partition [-]
    dry_threshold : xarray.DataArray
        Dry Threshold friction velocity [m/s]

    Returns
    -------
    xarray.DataArray
        Total mass emitted [g/s]
    """
    func = lambda a, b, c, d, e, f, g, h, i: pyfengsha.fengsha(a, b, c, d, e, f, g, h, i)
    return xr.apply_ufunc(func, rhoa, volumetric_soil_moisture, ssm, land, ustar, clayfrac, sandfrac, drag_partition, dry_threshold, vectorize=True)
