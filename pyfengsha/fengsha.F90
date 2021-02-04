subroutine fengsha_albedo(rho_phy,smois,ssm,xland,ust,clay,sand,rdrag,u_ts0,emis_dust)
  IMPLICIT NONE

  REAL, INTENT(OUT) :: emis_dust
  REAL, INTENT(IN) :: smois         ! volumetric soil moisture m3/m3
  REAL, INTENT(IN) :: ssm           ! sediment supply map
  REAL, INTENT(IN) :: xland,      & ! land=1 or water=0
       ust,        & ! friction velocity (m/s)
       clay,       & ! clay fractions
       sand,       & ! sand fraction
       rdrag,      & ! drag partition (1/m)
       u_ts0,      & ! dry threshold friction velocity (m/s)
       rho_phy       ! air density [kg/m3]

  REAL, PARAMETER :: cmb=1.0
  ! Local variables

  integer :: ilwi
  real, parameter :: g0 = 9.81 * 100 ! gravity
  real :: kvh
  real :: rhoa
  real :: u_ts
  real :: H
  real :: Q
  real :: ustar_albedo
!f2py intent(out) :: emis_dust
!f2py intent(in) :: smois
!f2py intent(in) :: ssm
!f2py intent(in) :: xland
!f2py intent(in) :: ust
!f2py intent(in) :: clay
!f2py intent(in) :: sand
!f2py intent(in) :: rdrag
!f2py intent(in) :: u_ts0
!f2py intent(in) :: rho_phy

  ! Don't do dust over water!!!
  ilwi = 0
  if (xland.eq.1) then
     ilwi = 1
  end if
  ! dont do where ssm says not possible
  if (ssm > 0) then
     ilwi = 1
  end if

  IF (ilwi .eq. 0) THEN
     emis_dust = 0.
  ELSE
     rhoa = rho_phy * 1.0E-3
     ! soil moisture correction
     call fecan_moisture_correction(smois,sand,clay, H)

     ! vertical to horizontal mass flux
     call MB95_kvh(clay,kvh)

     ! modified threshold velocity
     call  modified_threshold(u_ts0, H, rdrag, u_ts)

     ! horizontal mass flux
     ustar_albedo = ust * rdrag
     call fengsha_hflux(ustar_albedo,u_ts, Q)

     emis_dust = cmb * ssm * rho_phy / g0 * kvh * Q

  end if

  return
end subroutine fengsha_albedo

subroutine fengsha(rho_phy,smois,ssm,xland,ust,clay,sand,rdrag,u_ts0,emis_dust)
  IMPLICIT NONE

  REAL, INTENT(OUT) :: emis_dust
  REAL, INTENT(IN) :: smois         ! volumetric soil moisture m3/m3
  REAL, INTENT(IN) :: ssm           ! sediment supply map
  REAL, INTENT(IN) :: xland,      & ! land=1 or water=0
       ust,        & ! friction velocity (m/s)
       clay,       & ! clay fractions
       sand,       & ! sand fraction
       rdrag,      & ! drag partition (1/m)
       u_ts0,       & ! dry threshold friction velocity (m/s)
       rho_phy       ! air density [kg/m3]

  REAL, PARAMETER :: cmb=1.0
  ! Local variables

  integer :: ilwi
  real, parameter :: g0 = 9.81 * 100 ! gravity
  real :: kvh
  real :: rhoa
  real :: u_ts
  real :: H
  real :: Q

  ! ! threshold values
  ! conver=1.e-9
  ! converi=1.e9

  ! Don't do dust over water!!!
  ilwi = 0
  if (xland.eq.1) then
     ilwi = 1
  end if
  ! dont do where ssm says not possible
  if (ssm > 0) then
     ilwi = 1
  end if

  IF (ilwi .eq. 0) THEN
     emis_dust = 0.
  ELSE
     rhoa = rho_phy * 1.0E-3
     ! soil moisture correction
     call fecan_moisture_correction(smois,sand,clay, H)

     ! vertical to horizontal mass flux
     call MB95_kvh(clay,kvh)

     ! modified threshold velocity
     call  modified_threshold(u_ts0, H, rdrag, u_ts)

     ! horizontal mass flux
     call fengsha_hflux(ust,u_ts, Q)

     emis_dust = cmb * ssm * rho_phy / g0 * kvh * Q

  end if

  return

end subroutine fengsha

subroutine mackinnon_drag(z0,R)

  IMPLICIT NONE

  real, intent(in) :: z0
  real, intent(out) :: R
  real, parameter :: z0s = 1.0e-04 !Surface roughness for ideal bare surface [m]
  ! ------------------------------------------------------------------------
  ! Function: Calculates the MacKinnon et al. 2004 Drag Partition Correction
  !
  !   R = 1.0 - log(z0 / z0s) / log( 0.7 * (12255./z0s) ** 0.8)
  !
  !--------------------------------------------------------------------------
  ! Drag partition correction. See MacKinnon et al. (2004),
  !     doi:10.1016/j.geomorph.2004.03.009
  R = 1.0 - log(z0 / z0s) / log( 0.7 * (12255./z0s) ** 0.8)

  return

end subroutine mackinnon_drag


subroutine mb95_drag(z0,R)

  IMPLICIT NONE

  real, intent(in) :: z0
  real, intent(out) :: R
  real, parameter :: z0s = 1.0e-04 !Surface roughness for ideal bare surface [m]

  ! Drag partition correction. See MacKinnon et al. (2004),
  !     doi:10.1016/j.geomorph.2004.03.009
  !  R = 1.0 - log(z0 / z0s) / log( 0.7 * (12255./z0s) ** 0.8)

  ! Drag partition correction. See Marticorena et al. (1997),
  !     doi:10.1029/96JD02964

  R = 1.0 - log(z0 / z0s) / log( 0.7 * (10./z0s) ** 0.8)

  return

end subroutine mb95_drag


subroutine fengsha_hflux(ust,utst, Q)
  !---------------------------------------------------------------------
  ! Function: Calculates the Horizontal Saltation Flux, Q, and then
  !           calculates the vertical flux.
  !
  ! formula of Draxler & Gillette (2001) Atmos. Environ.
  ! F   =  K A (r/g) U* ( U*^2 - Ut*^2 )
  !
  ! where:
  !     F   = vertical emission flux  [g/m**2-s]
  !     K   = constant 2.0E-04                      [1/m]
  !     A   = 0~3.5  mean = 2.8  (fudge factor)
  !     U*  = friction velocity                     [m/s]
  !     Ut* = threshold friction velocity           [m/s]
  !
  !--------------------------------------------------------------------
  real, intent(in) :: ust ! friction velocity
  real, intent(in) ::utst ! threshold friction velocity

  real, intent(out) :: Q
  Q = max(0.,ust * (ust * ust - utst * utst))

  return

end subroutine fengsha_hflux


subroutine MB95_kvh(clay,kvh)
  !---------------------------------------------------------------------
  ! Function: Calculates the vertical to horizontal mass flux ratio.
  !
  ! formula of MB95
  ! kvh = 10.0**(13.4*clay-6.0)
  !
  ! where:
  !     kvh   = vertical to hoizontal mass flux ratio [-]
  !     clay  = fractional clay content [-]
  !
  !--------------------------------------------------------------------
  real, intent(in) :: clay ! fractional clay content [-]

  real, intent(out) :: kvh
  if (clay <= 0.2) then
     kvh=10.0**(13.4*clay-6.0)
  else
     kvh = 2.E-4
  endif

  return

end subroutine MB95_kvh

subroutine fecan_moisture_correction(vol_soil_moisture,sand,clay, H)
  !---------------------------------------------------------------------
  ! Function: calculates the fecan soil moisture
  ! drylimit = 14.0*clay*clay+17.0*clay
  ! H = sqrt(1.0 + 1.21*(gravsm-drylimit)**0.68)
  !---------------------------------------------------------------------
  real, intent(in) :: vol_soil_moisture ! fractional clay content [-]
  real, intent(in) :: sand ! fractional sand content [-]
  real, intent(in) :: clay ! fractional clay content [-]
  real, parameter :: soil_dens = 2650. ! soil density [kg/m3]
  real, parameter :: water_dens = 1000. ! water density [kg/m3]
  real :: GRAVSM
  real :: drylimit
  real, intent(out) :: H ! fecan soil moisture adjustment

  H = 0.

  call volumetric_to_gravimetric(vol_soil_moisture,sand,gravsm)

  ! fecan dry limit
  drylimit=14.0*clay*clay+17.0*clay

  ! fecan soil moisture correction
  IF (gravsm > drylimit) THEN
     H = sqrt(1.0 + 1.21*(gravsm-drylimit)**0.68)
  ELSE
     H = 1.0
  END IF

  return

end subroutine fecan_moisture_correction

subroutine shao_1996_soil_moisture(w, H)

  Implicit None

  ! inputs
  real, intent(in) :: w ! volumetric soil moisture [m3/m3]

  !outputs
  real, intent(out) :: H

  H = 0.

  H = exp(22.7 * w)

  return

end subroutine shao_1996_soil_moisture


subroutine shao_2004_soil_moisture(w, H)

  Implicit None

  ! inputs
  real, intent(in) :: w ! volumetric soil moisture [m3/m3]

  !outputs
  real, intent(out) :: H

  H = 0.
  if (w <= 0.03) then
     H = exp(22.7 * w)
  else
     H = exp(95.3 * w - 2.029)
  end if

  return

end subroutine shao_2004_soil_moisture

subroutine fecan_dry_limit(clay,drylimit)

  IMPLICIT NONE

  !Inputs
  real, intent(in) :: clay ! fractional clay content
  !outpus
  real, intent(out) :: drylimit ! fecan dry limit [kg/kg]

  drylimit = 0.
  if (clay <= 0) then
     drylimit = 14. * 1e-4 * 1e-4 + 17. * 1e-4
  else
     drylimit = 14 * clay * clay + 17 * clay
  end if

  return
end subroutine fecan_dry_limit

subroutine volumetric_to_gravimetric(vsoil, sandfrac, grav_soil)

  IMPLICIT NONE

  ! Inputs
  real, intent(in) :: vsoil ! Volumetric Soil Moisture [m3/m3]
  real, intent(in) :: sandfrac ! fractional Sand content
  ! outputs
  real, intent(out) :: grav_soil ! gravimetric soil moisture [kg/kg]
  ! Local
  real, parameter :: soil_dens = 2650.
  real, parameter :: water_dens = 1000.
  real :: vsat
  grav_soil = 0.
  vsat = 0.

  ! Saturated volumetric water content (sand-dependent) ! [m3 m-3]
  vsat = 0.489 - 0.00126 * ( sandfrac * 100 )
  ! gravimetric soil content
  grav_soil = vsoil * water_dens / (soil_dens * (1. - vsat))

  return
end subroutine volumetric_to_gravimetric

subroutine modified_threshold(u_ts0, H, drag, u_ts)

  IMPLICIT NONE

  real, intent(in) :: u_ts0 ! dry threshold velocity
  real, intent(in) :: H ! fecan soil moisture correction
  real, intent(in) :: drag ! drag partition
  real, intent(out) :: u_ts ! modified threshold velocity
  u_ts = 0.
  u_ts = u_ts0 * H / drag

  return
end subroutine modified_threshold
