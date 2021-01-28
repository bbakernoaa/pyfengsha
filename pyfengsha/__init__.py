import fengsha as fdust

def fengsha_albedo(rhoa,volumetric_soil_moisture, ssm, land, ustar, clayfrac, sandfrac, drag_partition, dry_threshold):
    emission = fdust.fengsha_albedo(rhoa,volumetric_soil_moisture, ssm, land, ustar, clayfrac, sandfrac, drag_partition, dry_threshold)
    return emission

def fengsha(rhoa,volumetric_soil_moisture, ssm, land, ustar, clayfrac, sandfrac, drag_partition, dry_threshold):
    emission = fdust.fengsha(rhoa,volumetric_soil_moisture, ssm, land, ustar, clayfrac, sandfrac, drag_partition, dry_threshold)
    return emission

def mackinnon_drag_partition(z0):
    return fdust.mackinnon_drag(z0)

def mb95_drag_partition(z0):
    return fdust.mb95_drag(z0)

def draxler_hflux(ustar, threshold_velocity):
    return fdust.fengsha_hflux(ustar,threshold_velocity)

def mb95_kvh(clay):
    return fdust.mb95_kvh(clay)

def fecan_moisture_correction(volumetric_soil_moisture,sandfrac,clayfrac):
    return fdust.fecan_moisture_correction(volumetric_soil_moisture,sandfrac,clayfrac)

def fecan_dry_limit(clayfrac):
    return fdust.fecan_dry_limit(clayfrac)

def volumetric_to_gravimetric(volumetric_soil_moisture,sandfrac):
    return fdust.volumetric_soil_moisture(volumetric_soil_moisture,sandfrac)

def shao_1996_soil_moisture(volumetric_soil_moisture):
    return fdust.shao_1996_soil_moisture(volumetric_soil_moisture)

def shao_2004_soil_moisture(volumetric_soil_moisture):
    return fdust.shao_2004_soil_moisture(volumetric_soil_moisture)

def modified_threshold_velocity(dry_threshold, moisture_correction, drag_partition):
    return fdust.modified_threshold(dry_threshold, moisture_correction, drag_partition)

